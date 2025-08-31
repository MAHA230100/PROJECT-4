import io
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


@st.cache_resource
def load_checkpoint(checkpoint_path: Path) -> Tuple[nn.Module, List[str], int]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    class_names: List[str] = ckpt.get("class_names", [])
    image_size: int = int(ckpt.get("image_size", 224))
    backbone: str = str(ckpt.get("backbone", "resnet18"))

    if backbone == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(class_names))
    elif backbone == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(class_names))
    elif backbone == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, len(class_names))
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, len(class_names))
    else:
        raise ValueError(f"Unsupported backbone in checkpoint: {backbone}")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, class_names, image_size


def build_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict_topk(model: nn.Module, image: Image.Image, tfm: transforms.Compose, class_names: List[str], k: int = 3) -> List[Tuple[str, float]]:
    with torch.no_grad():
        x = tfm(image.convert("RGB")).unsqueeze(0)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    topk_idx = np.argsort(probs)[::-1][:k]
    return [(class_names[i], float(probs[i])) for i in topk_idx]


def get_waste_category_color(category: str) -> str:
    """Return color for waste category visualization"""
    colors = {
        'plastic': '#1f77b4',      # Blue
        'glass': '#2ca02c',        # Green  
        'metal': '#ff7f0e',        # Orange
        'paper': '#d62728',        # Red
        'cardboard': '#9467bd',    # Purple
        'organic': '#8c564b',      # Brown
        'biological': '#8c564b',   # Brown
        'battery': '#e377c2',      # Pink
        'clothes': '#7f7f7f',      # Gray
        'shoes': '#bcbd22',        # Yellow-green
        'trash': '#17becf',        # Cyan
        'brown-glass': '#2ca02c',  # Green
        'green-glass': '#2ca02c',  # Green
        'white-glass': '#2ca02c',  # Green
    }
    return colors.get(category.lower(), '#1f77b4')

def is_recyclable(category: str) -> bool:
    """Check if waste category is recyclable"""
    recyclable = ['plastic', 'glass', 'metal', 'paper', 'cardboard', 'brown-glass', 'green-glass', 'white-glass']
    return category.lower() in recyclable

def load_sample_images(data_dir: Path, class_names: List[str], num_samples: int = 2) -> Dict[str, List[Image.Image]]:
    """Load sample images from each class for display"""
    samples = {}
    for class_name in class_names:
        class_dir = data_dir / class_name
        if class_dir.exists():
            image_files = list(class_dir.glob("*.jpg"))[:num_samples]
            samples[class_name] = []
            for img_path in image_files:
                try:
                    img = Image.open(img_path)
                    samples[class_name].append(img)
                except Exception:
                    continue
    return samples

def get_class_distribution(data_dir: Path, class_names: List[str]) -> Dict[str, int]:
    """Get the number of images per class"""
    distribution = {}
    for class_name in class_names:
        class_dir = data_dir / class_name
        if class_dir.exists():
            count = len(list(class_dir.glob("*.jpg")))
            distribution[class_name] = count
        else:
            distribution[class_name] = 0
    return distribution

def main() -> None:
    st.set_page_config(page_title="Garbage Classifier", page_icon="🗑️", layout="wide")
    st.title("Waste Classification System")
    st.markdown("**Upload an image to classify waste and get recycling guidance**")

    # Sidebar for model selection
    with st.sidebar:
        st.header("Model Settings")
        default_outputs = Path("outputs") / "resnet18"
        checkpoint_path_str = st.text_input("Path to model checkpoint (.pth)", value=str(default_outputs / "best_model.pth"))
        checkpoint_path = Path(checkpoint_path_str)

    # Main content area
    tab1, tab2, tab3 = st.tabs(["Dataset Analysis", "Classification", "Model Performance"])
    
    with tab1:
        st.header("Exploratory Data Analysis (EDA)")
        
        # Dataset path
        data_dir = Path("dataset/garbage_classification")
        if not data_dir.exists():
            st.error("Dataset not found at dataset/garbage_classification")
            return
            
        # Get class names from dataset
        class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
        
        # Class distribution
        st.subheader("Class Distribution")
        distribution = get_class_distribution(data_dir, class_names)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            # Bar chart
            import plotly.express as px
            df_dist = pd.DataFrame(list(distribution.items()), columns=['Class', 'Count'])
            fig = px.bar(df_dist, x='Class', y='Count', 
                        title="Number of Images per Class",
                        color='Count', color_continuous_scale='viridis')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Summary statistics
            st.markdown("### Summary")
            total_images = sum(distribution.values())
            st.metric("Total Images", f"{total_images:,}")
            st.metric("Number of Classes", len(class_names))
            st.metric("Average per Class", f"{total_images//len(class_names):,}")
            
            # Class balance info
            min_count = min(distribution.values())
            max_count = max(distribution.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else 0
            st.metric("Imbalance Ratio", f"{imbalance_ratio:.1f}x")
        
        # Sample images from each category
        st.subheader("Sample Images from Each Category")
        st.markdown("**Click on any image to see details**")
        
        samples = load_sample_images(data_dir, class_names, num_samples=3)
        
        # Create a grid of sample images
        cols = st.columns(4)
        for idx, class_name in enumerate(class_names):
            col_idx = idx % 4
            with cols[col_idx]:
                st.markdown(f"**{class_name.title()}**")
                if class_name in samples and samples[class_name]:
                    # Show first sample image
                    img = samples[class_name][0]
                    st.image(img, caption=f"{class_name} (n={distribution[class_name]})", 
                            use_container_width=True)
                    
                    # Show additional samples in expander
                    if len(samples[class_name]) > 1:
                        with st.expander(f"More {class_name} samples"):
                            for i, sample_img in enumerate(samples[class_name][1:], 1):
                                st.image(sample_img, caption=f"Sample {i}", use_container_width=True)
                else:
                    st.info(f"No images found for {class_name}")
        
        # Color distribution analysis
        st.subheader("Color Distribution Analysis")
        st.markdown("**Average RGB values per class**")
        
        # Calculate average RGB for each class
        color_data = []
        for class_name in class_names:
            if class_name in samples and samples[class_name]:
                avg_r, avg_g, avg_b = 0, 0, 0
                count = 0
                for img in samples[class_name]:
                    img_array = np.array(img)
                    if len(img_array.shape) == 3:
                        avg_r += img_array[:, :, 0].mean()
                        avg_g += img_array[:, :, 1].mean()
                        avg_b += img_array[:, :, 2].mean()
                        count += 1
                
                if count > 0:
                    color_data.append({
                        'Class': class_name,
                        'Red': avg_r / count,
                        'Green': avg_g / count,
                        'Blue': avg_b / count
                    })
        
        if color_data:
            df_color = pd.DataFrame(color_data)
            fig_color = px.scatter_3d(df_color, x='Red', y='Green', z='Blue', 
                                    color='Class', title="3D Color Distribution by Class")
            st.plotly_chart(fig_color, use_container_width=True)
    
    with tab2:
        st.header("Image Classification")
        
        if not checkpoint_path.exists():
            st.warning("Checkpoint not found. Train the model first (see README).")
            return

        model, class_names, image_size = load_checkpoint(checkpoint_path)
        tfm = build_transforms(image_size)

        uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            image = Image.open(io.BytesIO(uploaded.read()))
            st.image(image, caption="Uploaded Image", use_container_width=True)

            topk = predict_topk(model, image, tfm, class_names, k=3)
            
            # Primary prediction with emphasis
            primary_label, primary_prob = topk[0]
            primary_color = get_waste_category_color(primary_label)
            recyclable_status = "RECYCLABLE" if is_recyclable(primary_label) else "NOT RECYCLABLE"
            
            st.markdown("---")
            st.markdown("## **Primary Classification**")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"""
                <div style="background-color: {primary_color}; padding: 20px; border-radius: 10px; text-align: center; color: white;">
                    <h2>{primary_label.upper()}</h2>
                    <h3>{primary_prob*100:.1f}%</h3>
                    <p>{recyclable_status}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### Waste Category Details")
                if primary_label.lower() in ['plastic', 'glass', 'metal', 'paper', 'cardboard']:
                    st.success(f"**{primary_label.title()}** can be recycled!")
                    st.info("**Recycling Tip**: Clean and separate by type for better processing.")
                elif primary_label.lower() in ['organic', 'biological']:
                    st.warning(f"**{primary_label.title()}** should be composted.")
                    st.info("**Composting Tip**: Add to organic waste bin or home compost.")
                elif primary_label.lower() in ['battery']:
                    st.error(f"**{primary_label.title()}** requires special disposal.")
                    st.info("**Disposal Tip**: Take to battery recycling center or electronics store.")
                else:
                    st.info(f"**{primary_label.title()}** may need special handling.")
                    st.info("**Check local guidelines** for proper disposal.")

            # All predictions
            st.markdown("##**All Predictions**")
            for i, (label, prob) in enumerate(topk):
                color = get_waste_category_color(label)
                recyclable = "♻️" if is_recyclable(label) else "⚠️"
                confidence_bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
                
                st.markdown(f"""
                <div style="border-left: 5px solid {color}; padding-left: 10px; margin: 5px 0;">
                    <strong>{label.title()}</strong> {recyclable}<br>
                    <small>{confidence_bar} {prob*100:.1f}%</small>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.header("Model Performance")
        
        if not checkpoint_path.exists():
            st.warning("Checkpoint not found. Train the model first (see README).")
            return
            
        # Load training history if available
        history_path = checkpoint_path.parent / "training_history.json"
        if history_path.exists():
            import json
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            st.subheader("Training Progress")
            
            # Training curves
            epochs = range(1, len(history.get("train_loss", [])) + 1)
            
            col1, col2 = st.columns(2)
            with col1:
                fig_loss = px.line(x=epochs, y=history.get("train_loss", []), 
                                 title="Training Loss", labels={'x': 'Epoch', 'y': 'Loss'})
                if "val_loss" in history:
                    # fig_loss.add_scatter(x=epochs, y=history["val_loss"], name="Validation Loss")

                    epochs = list(range(1, len(history["val_loss"]) + 1))
                    fig_loss.add_scatter(
                            x=epochs,
                            y=history["val_loss"],
                            name="Validation Loss"
                            )
                st.plotly_chart(fig_loss, use_container_width=True)
            
            with col2:
                fig_acc = px.line(x=epochs, y=history.get("train_acc", []), 
                                title="Training Accuracy", labels={'x': 'Epoch', 'y': 'Accuracy'})
                if "val_acc" in history:
                    fig_acc.add_scatter(x=epochs, y=history["val_acc"], name="Validation Accuracy")
                st.plotly_chart(fig_acc, use_container_width=True)
            
            # F1 score if available
            if "val_f1_macro" in history:
                fig_f1 = px.line(x=epochs, y=history["val_f1_macro"], 
                               title="Validation F1 Score (Macro)", 
                               labels={'x': 'Epoch', 'y': 'F1 Score'})
                st.plotly_chart(fig_f1, use_container_width=True)
        
        # Load classification report if available
        report_path = checkpoint_path.parent / "classification_report.csv"
        if report_path.exists():
            st.subheader("Classification Report")
            df_report = pd.read_csv(report_path, index_col=0)
            st.dataframe(df_report, use_container_width=True)
        
        # Load confusion matrix if available
        cm_path = checkpoint_path.parent / "confusion_matrix.csv"
        if cm_path.exists():
            st.subheader("Confusion Matrix")
            cm_df = pd.read_csv(cm_path, index_col=0)
            
            # Create heatmap
            fig_cm = px.imshow(cm_df.values, 
                              x=cm_df.columns, y=cm_df.index,
                              title="Confusion Matrix",
                              color_continuous_scale='Blues',
                              aspect="auto")
            st.plotly_chart(fig_cm, use_container_width=True)


if __name__ == "__main__":
    main()


