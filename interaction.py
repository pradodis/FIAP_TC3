import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json

class ModelTester:
    def __init__(self, base_model_path, lora_adapters_dir, dataset_path=None):
        """
        Initialize the model tester with base model and LoRA adapters
        
        Args:
            base_model_path: Path to the base model
            lora_adapters_dir: Directory containing LoRA adapter folders
            dataset_path: Path to training dataset for RAG (JSONL format)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Load tokenizer and base model with quantization
        print("Loading base model with 4-bit quantization...")
        self.tokenizer, self.base_model = self.load_model_with_quantization(base_model_path)
        
        # Load only final LoRA adapter
        self.final_lora_model = None
        self.load_final_lora_adapter(lora_adapters_dir)
        
        # Initialize RAG if dataset provided
        self.rag_enabled = False
        if dataset_path and os.path.exists(dataset_path):
            self.vectorizer, self.tfidf_matrix, self.titles, self.documents, self.structured_data = self.build_rag_index(dataset_path)
            self.rag_enabled = True
            print(f"✓ RAG index built from {len(self.documents)} documents")
        else:
            print("⚠ No dataset provided - RAG disabled")
    
    def load_model_with_quantization(self, model_path):
        """
        Load Mistral model with 4-bit quantization (same as fine-tuning script)
        """
        # 4-bit quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.uint8
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load model with same configuration as fine-tuning
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        
        return tokenizer, model
    
    def load_final_lora_adapter(self, lora_adapters_dir):
        """Load only the final LoRA adapter"""
        print("Loading final LoRA adapter...")
        
        if os.path.exists(lora_adapters_dir):
            adapter_files = ['adapter_config.json', 'adapter_model.bin', 'adapter_model.safetensors']
            if any(os.path.exists(os.path.join(lora_adapters_dir, f)) for f in adapter_files):
                try:
                    self.final_lora_model = PeftModel.from_pretrained(self.base_model, lora_adapters_dir)
                    print(f"✓ Final LoRA adapter loaded successfully from {lora_adapters_dir}")
                except Exception as e:
                    print(f"✗ Failed to load final LoRA adapter: {e}")
            else:
                print(f"✗ No adapter files found in {lora_adapters_dir}")
        else:
            print(f"✗ Directory not found: {lora_adapters_dir}")
    
    def format_mistral_prompt(self, prompt, instruction=None):
        """
        Format prompt in Mistral-Instruct style (matching fine-tuning script)
        """
        if instruction is None:
            instruction = "Describe the product concisely and in a human-like manner."
        
        # Match the format used in fine-tuning
        return f"<s>[INST] {instruction}\n\nProduto: {prompt} [/INST]"
    
    def generate_text(self, prompt, model, max_length=100, temperature=0.7, top_p=0.9):
        """Generate text using the specified model with 100-word limit"""
        # Format prompt for Mistral if it looks like a simple product name
        if len(prompt.split()) <= 5 and not prompt.startswith("Gere"):
            formatted_prompt = self.format_mistral_prompt(prompt)
        else:
            formatted_prompt = prompt
            
        inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=150,  # Limited tokens for ~100 words
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                num_return_sequences=1
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from the output
        response = generated_text[len(self.tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()
        
        # Limit to approximately 100 words
        words = response.split()
        if len(words) > 100:
            response = ' '.join(words[:100]) + '...'
        
        return response
    
    def build_rag_index(self, dataset_path):
        """Build TF-IDF index for RAG from the training dataset (JSONL format)"""
        print("Building RAG index from structured dataset...")
        
        # Load structured dataset
        structured_data = []
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        structured_data.append(json.loads(line))
            
            print(f"Loaded {len(structured_data)} structured records")
        except Exception as e:
            print(f"Error loading structured dataset: {e}")
            # Fallback to old format
            return self.build_rag_index_fallback(dataset_path)
        
        # Sample for performance (if too large)
        if len(structured_data) > 15000:
            import random
            random.seed(42)
            structured_data = random.sample(structured_data, 15000)
            print(f"Sampled down to {len(structured_data)} records for performance")
        
        # Prepare documents from structured data
        documents = []
        titles = []
        
        for item in structured_data:
            input_text = item.get('input', '')
            output_text = item.get('output', '')
            instruction = item.get('instruction', '')
            
            if input_text and output_text:
                # Combine input and output for comprehensive search
                doc = f"{input_text} {output_text}"
                documents.append(doc)
                titles.append(input_text)
        
        if not documents:
            print("No valid documents found in structured data")
            return None, None, [], [], []
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.85,
            stop_words='english'
        )
        
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        return vectorizer, tfidf_matrix, titles, documents, structured_data
    
    def build_rag_index_fallback(self, dataset_path):
        """Fallback RAG index for old format"""
        print("Using fallback RAG index for old format...")
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        df = pd.DataFrame(data)
        df = df[['title', 'content']].dropna()
        
        # Sample for performance (if too large)
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42)
        
        # Prepare documents
        documents = []
        titles = []
        
        for _, row in df.iterrows():
            doc = f"{row['title']} {row['content']}"
            documents.append(doc)
            titles.append(row['title'])
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
        
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        return vectorizer, tfidf_matrix, titles, documents, []
    
    def retrieve_context(self, query, top_k=3):
        """Retrieve relevant context for RAG with improved scoring"""
        if not self.rag_enabled:
            return []
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k results with minimum threshold
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        context_items = []
        for idx in top_indices:
            if similarities[idx] > 0.05:  # Lower threshold for better recall
                context_info = {
                    'title': self.titles[idx],
                    'snippet': self.documents[idx][:400],  # Shorter snippets
                    'score': similarities[idx]
                }
                
                # Add structured data if available
                if hasattr(self, 'structured_data') and self.structured_data and idx < len(self.structured_data):
                    struct_item = self.structured_data[idx]
                    context_info['instruction'] = struct_item.get('instruction', '')
                    context_info['full_output'] = struct_item.get('output', '')
                
                context_items.append(context_info)
        
        return context_items
    
    def format_prompt_with_rag(self, prompt, instruction=None):
        """Format prompt with RAG context using structured data"""
        context_items = self.retrieve_context(prompt)
        
        if not context_items:
            return self.format_mistral_prompt(prompt, instruction), []
        
        # Build enhanced context from structured data
        context_examples = []
        for item in context_items:
            if 'full_output' in item and item['full_output']:
                context_examples.append(f"Exemplo: {item['title']} -> {item['full_output'][:200]}")
            else:
                context_examples.append(f"Contexto: {item['snippet'][:200]}")
        
        context_text = "\n".join(context_examples[:2])  # Limit to 2 examples
        
        # Enhanced instruction with context
        enhanced_instruction = f"""Baseando-se nos exemplos fornecidos, {instruction or 'descreva o produto de forma detalhada e atrativa'}:

{context_text}

Agora, seguindo o mesmo padrão dos exemplos acima"""
        
        enhanced_prompt = self.format_mistral_prompt(prompt, enhanced_instruction)
        
        return enhanced_prompt, context_items
    
    def test_final_lora_with_rag(self, prompt, max_length=100, temperature=0.7, top_p=0.9, instruction=None):
        """Test only the final LoRA model with RAG"""
        if not self.final_lora_model:
            print("❌ Final LoRA model not available")
            return
            
        print("\n" + "="*80)
        print(f"PROMPT: {prompt}")
        print("="*80)
        
        if self.rag_enabled:
            enhanced_prompt, context_items = self.format_prompt_with_rag(prompt, instruction)
            
            if context_items:
                print(f"\n🟢 FINAL LORA COM RAG:")
                print("="*50)
                print("📚 CONTEXTO RECUPERADO:")
                for i, item in enumerate(context_items, 1):
                    print(f"  {i}. {item['title']} (score: {item['score']:.3f})")
                print()
                
                # Generate response with final LoRA model
                try:
                    response = self.generate_text(enhanced_prompt, self.final_lora_model, max_length, temperature, top_p)
                    print("🤖 RESPOSTA:")
                    print(response)
                    
                    word_count = len(response.split())
                    print(f"\n📊 Palavras: {word_count}/100 | Caracteres: {len(response)}")
                    
                    # Show sources
                    print("\n📖 FONTES UTILIZADAS:")
                    for item in context_items:
                        print(f"  - {item['title']}")
                        
                except Exception as e:
                    print(f"❌ Erro ao gerar resposta: {e}")
            else:
                print(f"\n⚠ Nenhum contexto relevante encontrado para: '{prompt}'")
                print("Gerando resposta sem contexto...")
                
                formatted_prompt = self.format_mistral_prompt(prompt, instruction)
                try:
                    response = self.generate_text(formatted_prompt, self.final_lora_model, max_length, temperature, top_p)
                    print("🤖 RESPOSTA (sem RAG):")
                    print(response)
                    
                    word_count = len(response.split())
                    print(f"\n📊 Palavras: {word_count}/100 | Caracteres: {len(response)}")
                except Exception as e:
                    print(f"❌ Erro ao gerar resposta: {e}")
        else:
            print(f"\n🔵 FINAL LORA SEM RAG:")
            print("="*50)
            formatted_prompt = self.format_mistral_prompt(prompt, instruction)
            try:
                response = self.generate_text(formatted_prompt, self.final_lora_model, max_length, temperature, top_p)
                print("🤖 RESPOSTA:")
                print(response)
                
                word_count = len(response.split())
                print(f"\n📊 Palavras: {word_count}/100 | Caracteres: {len(response)}")
            except Exception as e:
                print(f"❌ Erro ao gerar resposta: {e}")
    
    def interactive_session(self):
        """Simplified interactive session with only final LoRA + RAG"""
        print("\n🚀 INTERACTIVE SESSION - FINAL LORA + RAG")
        print("="*60)
        print(f"Model: Mistral-7B-Instruct-v0.3 (4-bit quantized)")
        print(f"Final LoRA: {'✓ Loaded' if self.final_lora_model else '❌ Not available'}")
        print(f"RAG enabled: {'✓ Yes' if self.rag_enabled else '✗ No'}")
        print(f"Response limit: 100 words maximum")
        
        print("\nCommands:")
        print("  - Type your product prompt and press Enter")
        print("  - Type 'instruct [instruction] | [prompt]' for custom instruction")
        print("  - Type 'quit' or 'exit' to end session")
        print("  - Type 'settings' to adjust generation parameters")
        print("  - Type 'examples' to see example prompts")
        print("="*60)
        
        if not self.final_lora_model:
            print("❌ Cannot start session - Final LoRA model not loaded")
            return
        
        # Default settings optimized for 100-word responses
        max_length = 100
        temperature = 0.7
        top_p = 0.9
        
        while True:
            try:
                user_input = input("\nEnter your prompt: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'examples':
                    print("\n📝 Example prompts:")
                    print("  - Smartwatch Fitness Tracker")
                    print("  - Notebook gamer com RGB")
                    print("  - Câmera digital profissional")
                    print("  - Fones bluetooth premium")
                    print("  - instruct Crie uma descrição técnica | Processador Intel i7")
                    continue
                
                if user_input.lower() == 'settings':
                    print(f"\nCurrent settings:")
                    print(f"  Max words: {max_length}")
                    print(f"  Temperature: {temperature}")
                    print(f"  Top-p: {top_p}")
                    
                    try:
                        new_max_length = input(f"Max words (1-150, current {max_length}): ").strip()
                        if new_max_length:
                            max_length = min(150, max(1, int(new_max_length)))
                        
                        new_temperature = input(f"Temperature (0.1-1.0, current {temperature}): ").strip()
                        if new_temperature:
                            temperature = max(0.1, min(1.0, float(new_temperature)))
                        
                        new_top_p = input(f"Top-p (0.1-1.0, current {top_p}): ").strip()
                        if new_top_p:
                            top_p = max(0.1, min(1.0, float(new_top_p)))
                        
                        print("✓ Settings updated!")
                    except ValueError:
                        print("Invalid input. Settings unchanged.")
                    continue
                
                if not user_input:
                    continue
                
                # Parse command
                instruction = None
                if user_input.lower().startswith('instruct '):
                    parts = user_input[9:].split(' | ', 1)
                    if len(parts) == 2:
                        instruction, prompt = parts
                        print(f"\n🎯 INSTRUÇÃO CUSTOMIZADA: {instruction}")
                        self.test_final_lora_with_rag(prompt, max_length, temperature, top_p, instruction)
                    else:
                        print("⚠ Formato: instruct [instrução] | [prompt]")
                    continue
                
                # Default: test final LoRA with RAG
                self.test_final_lora_with_rag(user_input, max_length, temperature, top_p, instruction)
                
            except KeyboardInterrupt:
                print("\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function"""
    BASE_MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.3"
    LORA_ADAPTERS_DIR = "d:/FIAP_TC3/finetuned_lora_adapter"
    DATASET_PATH = "d:/FIAP_TC3/processed_dataset/train.jsonl"
    
    print("🤖 Mistral Final LoRA + RAG Tester (100-word limit)")
    print("="*60)
    
    try:
        tester = ModelTester(BASE_MODEL_PATH, LORA_ADAPTERS_DIR, DATASET_PATH)
        tester.interactive_session()
    except Exception as e:
        print(f"Failed to initialize: {e}")

if __name__ == "__main__":
    main()
