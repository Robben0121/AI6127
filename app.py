import gradio as gr
from hybrid_retriever import hybrid_search

def search_fn(query):
    results, latency = hybrid_search(query, top_k=5)
    out_text = f"Query latency: {latency} ms\n\n==== Top 5 Retrieved Passages ====\n"
    for idx, item in enumerate(results):
        out_text += f"\nRank {idx+1} | ASIN: {item['parent_asin']}\n"
        out_text += f"{item['passage']}\n"
    return out_text

with gr.Blocks(title="Product RAG Hybrid Retriever") as demo:
    gr.Markdown("# Hybrid RAG Search (BM25 + FAISS + RRF)")
    query = gr.Textbox(label="Input your product query", placeholder="e.g. waterproof men's boots")
    output = gr.Textbox(label="Retrieval Result", lines=12)
    btn = gr.Button("Search")
    btn.click(search_fn, inputs=[query], outputs=[output])

if __name__ == "__main__":
    demo.launch()
