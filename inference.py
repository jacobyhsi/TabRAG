import argparse
import os
import numpy as np
from src.embedder import SentenceTransformerEmbedder, VLLMEmbedder
from src.vector_store import VectorStore
from src.llm import VLLMLLMClient, HFLLMClient, OpenAILLMClient

def run_interactive_rag(args):
    print_retrieved = args.print_retrieved
    print(f"Loading vector index from {args.data_path}...")

    storage_path = os.path.join(args.data_path, "docstore")
    index = VectorStore.load(storage_path)
    if args.use_vllm:
        print(f'Using VLLM model: {args.model} at {args.vllm_ip}:{args.vllm_port}')
        model = VLLMLLMClient(model=args.model, ip=args.vllm_ip, port=args.vllm_port)
        embedder = VLLMEmbedder(args.embedder, tensor_parallel_size=1, gpu_memory_utilization=0.6)
    elif args.use_hf:
        print(f"Using HuggingFace model: {args.model}")
        model = HFLLMClient(model=args.model)
        embedder = SentenceTransformerEmbedder(args.embedder)
    elif args.use_openai:
        print(f"Using OpenAI model: {args.model}")
        model = OpenAILLMClient(model=args.model)
        embedder = SentenceTransformerEmbedder(args.embedder)
    else:
        print("Error: You must use one of HuggingFace or VLLM as a VLM inference provider.")
        return

    
    print("\n" + "="*50)
    print("TABRAG INTERACTIVE RAG (ICL ENABLED)")
    print("="*50 + "\n")

    while True:
        query_text = input("Query > ")
        if query_text.lower() in ['exit', 'quit', 'q']:
            break
            
        # retrieval
        q_embed = embedder.encode([query_text]).astype("float32")
        results = index.search(q_embed, k=3)
        
        # build Context for ICL
        context_blocks = []
        if print_retrieved:
            print("\n[Retrieved Context]")
        for i, hit in enumerate(results):
            meta = hit.get('meta', {})
            doc_id = meta.get('doc_id', 'Unknown')
            page = meta.get('page', 'N/A')
            text = hit.get('text', '')
            
            context_blocks.append(f"Source [{i+1}] (Doc: {doc_id}, Page: {page}):\n{text}")
            if print_retrieved:
                print(f"- {doc_id} (Pg {page}): {text[:100]}...")

        full_context = "\n\n".join(context_blocks)
        
        # generation with icl prompt
        prompt = f"""You are a research assistant. Use the following retrieved snippets to answer the user's question accurately. 
If the information is not in the context, say you don't know.

Context:
{full_context}

Question: {query_text}
Answer:"""

        print("\n[Answer]")
        response = model.generate(system_message="", user_message=prompt)
        print(response)
        
        print("-" * 50 + "\n")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str, required=True, help="Retrieval storage path")
    argparser.add_argument("--embedder", type=str, required=True, default="Qwen/Qwen3-Embedding-0.6B")
    argparser.add_argument("--model", type=str, required=True, default="Qwen/Qwen3-1.7B")

    argparser.add_argument("--use_openai", action='store_true')

    argparser.add_argument("--use_hf", action='store_true')

    argparser.add_argument("--use_vllm", action='store_true')
    argparser.add_argument("--vllm_ip", type=str, help="IP address for VLLM server")
    argparser.add_argument("--vllm_port", type=str, help="Port for VLLM server")

    argparser.add_argument("--print_retrieved", action='store_true', help="Whether to print retrieved context snippets")
    args = argparser.parse_args()

    run_interactive_rag(args)
    