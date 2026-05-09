"""
セマンティック画像検索 — Gradio WebUI
"""

import gradio as gr
from PIL import Image

from src.config import AVAILABLE_MODELS, DEFAULT_MODEL, DEVICE
from src.embedder import get_embedder
from src.store import ImageStore

print(f"Using device: {DEVICE}")
print(f"モデルをロード中: {DEFAULT_MODEL} ...")
embedder = get_embedder(DEFAULT_MODEL)
store = ImageStore(model_key=DEFAULT_MODEL)
print("モデルロード完了")


PAGE_SIZE = 10
MAX_RESULTS = 100


def _fetch_results(emb) -> list:
    count = store.count()
    if count == 0:
        return []
    results = store.query(emb, n_results=min(MAX_RESULTS, count))
    output = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        similarity = 1 - dist
        img = Image.open(meta["path"]).convert("RGB")
        label = f"類似度: {similarity:.2%}\n{meta['caption']}"
        output.append((img, label))
    return output


def switch_model(model_key: str):
    global embedder, store
    print(f"モデルを切り替え中: {model_key} ...")
    embedder = get_embedder(model_key)
    store = ImageStore(model_key=model_key)
    count = store.count()
    if count == 0:
        msg = (
            f"⚠️ '{model_key}' のインデックスが空です。"
            f" 先に `python scripts/index_kitti.py --model {model_key}` を実行してください。"
        )
    else:
        msg = f"✓ モデル: {model_key}  （インデックス: {count:,} 件）"
    print(msg)
    return msg


def text_search_init(query: str):
    if not query.strip():
        return [], [], 0, gr.update(visible=False)
    if store.count() == 0:
        return [], [], 0, gr.update(visible=False)

    emb = embedder.get_text_embedding(query)
    all_results = _fetch_results(emb)
    shown = all_results[:PAGE_SIZE]
    has_more = len(all_results) > PAGE_SIZE
    return shown, all_results, PAGE_SIZE, gr.update(visible=has_more)


def image_search_init(query_image):
    if query_image is None:
        return [], [], 0, gr.update(visible=False)
    if store.count() == 0:
        return [], [], 0, gr.update(visible=False)

    image = Image.fromarray(query_image).convert("RGB")
    emb = embedder.get_image_embedding(image)
    all_results = _fetch_results(emb)
    shown = all_results[:PAGE_SIZE]
    has_more = len(all_results) > PAGE_SIZE
    return shown, all_results, PAGE_SIZE, gr.update(visible=has_more)


def load_more(all_results: list, current_count: int):
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
    **CLIP / SigLIP + ChromaDB** を使ったテキスト・画像検索のデモ
    """, elem_classes="header")

    # ---- モデル選択 ----
    with gr.Row():
        model_selector = gr.Radio(
            choices=AVAILABLE_MODELS,
            value=DEFAULT_MODEL,
            label="埋め込みモデル",
            info="モデルを切り替えると、そのモデルでインデックス済みの画像を検索します。",
        )
        model_status = gr.Textbox(
            value=f"✓ モデル: {DEFAULT_MODEL}  （インデックス: {store.count():,} 件）",
            label="ステータス",
            interactive=False,
        )

    model_selector.change(
        fn=switch_model,
        inputs=[model_selector],
        outputs=[model_status],
    )

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
    **仕組み**: 画像は事前に各モデルでベクトル化してChromaDBに保存。検索時はテキスト/画像クエリをベクトル化し、コサイン類似度で最近傍を返します。モデルごとに別のコレクションを使用します。
    """, elem_classes="subtext")

if __name__ == "__main__":
    demo.launch(share=False)
