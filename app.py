"""
CLIP + ChromaDB によるセマンティック画像検索
Step 3: Gradio WebUI版
"""

import gradio as gr
from PIL import Image

from src.config import DEVICE
from src.embedder import CLIPEmbedder
from src.store import ImageStore

print(f"Using device: {DEVICE}")
print("CLIPモデルをロード中...")
embedder = CLIPEmbedder()
store = ImageStore()
print("モデルロード完了")


PAGE_SIZE = 10
MAX_RESULTS = 100


def _fetch_results(emb) -> list:
    results = store.query(emb, n_results=MAX_RESULTS)
    output = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        similarity = 1 - dist
        img = Image.open(meta["path"]).convert("RGB")
        label = f"類似度: {similarity:.2%}\n{meta['caption']}"
        output.append((img, label))
    return output


def text_search_init(query: str):
    """テキストクエリで画像を検索（初回）"""
    if not query.strip():
        return [], [], 0, gr.update(visible=False)

    emb = embedder.get_text_embedding(query)
    all_results = _fetch_results(emb)
    shown = all_results[:PAGE_SIZE]
    has_more = len(all_results) > PAGE_SIZE
    return shown, all_results, PAGE_SIZE, gr.update(visible=has_more)


def image_search_init(query_image):
    """画像クエリで類似画像を検索（初回）"""
    if query_image is None:
        return [], [], 0, gr.update(visible=False)

    image = Image.fromarray(query_image).convert("RGB")
    emb = embedder.get_image_embedding(image)
    all_results = _fetch_results(emb)
    shown = all_results[:PAGE_SIZE]
    has_more = len(all_results) > PAGE_SIZE
    return shown, all_results, PAGE_SIZE, gr.update(visible=has_more)


def load_more(all_results: list, current_count: int):
    """次のページを追加表示"""
    new_count = current_count + PAGE_SIZE
    shown = all_results[:new_count]
    has_more = len(all_results) > new_count
    return shown, new_count, gr.update(visible=has_more)


# ---- Gradio UI ----
with gr.Blocks(
    title="🔍 セマンティック画像検索",
    theme=gr.themes.Soft(),
    css="""
    .header { text-align: center; padding: 20px 0; }
    .subtext { color: #666; font-size: 0.9em; }
    .gradio-container { max-width: 1400px !important; margin: 0 auto !important; padding: 0 40px !important; }
    [data-testid="gallery"] { height: auto !important; max-height: none !important; overflow: visible !important; }
    [data-testid="gallery"] > div { height: auto !important; max-height: none !important; overflow: visible !important; }
    """
) as demo:
    gr.Markdown("""
    # 🔍 セマンティック画像検索
    **CLIP + ChromaDB** を使ったテキスト・画像検索のデモ
    """, elem_classes="header")

    with gr.Tabs():
        # --- テキスト検索タブ ---
        with gr.TabItem("📝 テキストで検索"):
            gr.Markdown("英語でシーンを説明すると、類似する画像を返します。", elem_classes="subtext")
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("検索クエリ（英語）")
                    text_input = gr.Textbox(
                        label="",
                        show_label=False,
                        placeholder="例: a dog running in a park",
                        lines=2
                    )
                    search_btn = gr.Button("🔍 検索", variant="primary")

                    gr.Markdown("**クエリ例:**")
                    gr.Examples(
                        examples=[
                            ["a car driving on the road"],
                            ["pedestrian walking on sidewalk"],
                            ["intersection with traffic"],
                            ["highway with multiple lanes"],
                            ["parked cars on street"],
                            ["urban street scene"],
                        ],
                        inputs=text_input
                    )

            text_all_results = gr.State([])
            text_shown_count = gr.State(0)
            text_results = gr.Gallery(
                label="検索結果",
                columns=5,
                height=None,
                show_label=True
            )
            text_more_btn = gr.Button("さらに10件表示", visible=False)

            search_btn.click(
                fn=text_search_init,
                inputs=[text_input],
                outputs=[text_results, text_all_results, text_shown_count, text_more_btn]
            )
            text_input.submit(
                fn=text_search_init,
                inputs=[text_input],
                outputs=[text_results, text_all_results, text_shown_count, text_more_btn]
            )
            text_more_btn.click(
                fn=load_more,
                inputs=[text_all_results, text_shown_count],
                outputs=[text_results, text_shown_count, text_more_btn]
            )

        # --- 画像検索タブ ---
        with gr.TabItem("🖼️ 画像で検索"):
            gr.Markdown("画像をアップロードすると、似ている画像を返します。", elem_classes="subtext")
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label="クエリ画像をアップロード")
                    img_search_btn = gr.Button("🔍 類似画像を検索", variant="primary")

            img_all_results = gr.State([])
            img_shown_count = gr.State(0)
            img_results = gr.Gallery(
                label="検索結果",
                columns=5,
                height=None,
                show_label=True
            )
            img_more_btn = gr.Button("さらに10件表示", visible=False)

            img_search_btn.click(
                fn=image_search_init,
                inputs=[image_input],
                outputs=[img_results, img_all_results, img_shown_count, img_more_btn]
            )
            img_more_btn.click(
                fn=load_more,
                inputs=[img_all_results, img_shown_count],
                outputs=[img_results, img_shown_count, img_more_btn]
            )

    gr.Markdown("""
    ---
    **仕組み**: 画像は事前にCLIPでベクトル化してChromaDBに保存。検索時はテキスト/画像クエリをベクトル化し、コサイン類似度で最近傍を返します。
    """, elem_classes="subtext")

if __name__ == "__main__":
    demo.launch(share=False)
