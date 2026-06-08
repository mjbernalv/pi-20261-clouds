from backend.visualization import plot_segmentation
from backend.inference import predict
import streamlit as st

def _segmentation_page() -> None:
    with st.sidebar:
        st.markdown(f"**User:** {st.session_state.username}")
        if st.button("Sign out"):
            st.session_state.logged_in = False
            st.rerun()
        st.divider()

    st.title("Satellite Image Segmentation")
    st.markdown(
            "Upload a satellite image and the model will detect cloud patterns: "
            "**Fish**, **Flower**, **Gravel**, and **Sugar**."
        )

    uploaded = st.file_uploader(
        "Upload a satellite image (.jpg or .png)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.info("Upload an image to get started.")
        return

    image_bytes = uploaded.read()

    left, right = st.columns(2)
    with left:
        st.subheader("Original image")
        st.image(image_bytes)

    with right:
        st.subheader("Segmentation result")
        with st.spinner("Running inference…"):
            try:
                result = predict(image_bytes)
            except FileNotFoundError as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                return

        fig = plot_segmentation(image_bytes, result)
        st.pyplot(fig, use_container_width=True)

    st.divider()
    st.subheader(f"Predictions  —  inference time: {result['inference_ms']} ms")
    total_pixels = result["width"] * result["height"]
    cols = st.columns(4)
    for col, cls in zip(cols, ["Fish", "Flower", "Gravel", "Sugar"]):
        pred = result["predictions"][cls]
        pct = pred["pixel_count"] / total_pixels * 100
        col.metric(cls, f"{pct:.1f}%", f"{pred['pixel_count']:,} px")
