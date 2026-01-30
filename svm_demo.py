import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_circles, make_moons
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

st.set_page_config(page_title="SEAS-8505 SVM Demo", layout="wide", initial_sidebar_state="expanded")

# Enhanced modern CSS styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }

    .block-container {
        background-color: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }

    /* Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 15px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background-color: white;
        border-radius: 12px;
        padding: 15px 25px;
        font-weight: 600;
        font-size: 16px;
        color: #495057;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
        transform: translateY(-2px);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: 2px solid #5a67d8;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    /* Modern Sliders */
    .stSlider > div > div > div {
        background-color: #e9ecef;
        border-radius: 10px;
    }

    .stSlider > div > div > div > div {
        background-color: #667eea;
        border-radius: 10px;
    }

    .stSlider [role="slider"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        width: 24px;
        height: 24px;
        border-radius: 50%;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.5);
        transition: all 0.2s ease;
    }

    .stSlider [role="slider"]:hover {
        transform: scale(1.2);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }

    /* Title Styling */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    h2 {
        color: #495057;
        font-weight: 600;
        margin-top: 2rem;
    }

    h3 {
        color: #667eea;
        font-weight: 600;
    }

    /* Modern Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }

    /* Info boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid #667eea;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }

    /* Select boxes */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
    }

    .stSelectbox > div > div:hover {
        border-color: #667eea;
    }

    /* Checkboxes */
    .stCheckbox {
        font-weight: 500;
    }

    /* Example boxes */
    .example-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 15px 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }

    .formula-box {
        background: linear-gradient(135deg, #fff5f5 0%, #ffe5e5 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #f56565;
        margin: 15px 0;
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
    }

    /* Dataframes */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

# Header with course info
st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px; margin-bottom: 30px; color: white; box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);'>
        <h1 style='color: white; -webkit-text-fill-color: white; margin: 0; font-size: 2.5rem;'>Support Vector Machine Interactive Demo</h1>
        <h3 style='color: white; margin: 10px 0; font-weight: 400;'>SEAS-8505 | Dr. Elbasheer | 1/17/2026</h3>
        <p style='font-size: 1.1rem; margin: 10px 0; opacity: 0.95;'>Learn SVM through Interactive Visualizations & Examples</p>
    </div>
    """, unsafe_allow_html=True)

tabs = st.tabs(["🎓 Start Here: Vectors", "📖 SVM Basics", "Interactive SVM", "🧮 Simple Example", "🔍 Constraints & Formulation", "Optimization", "📊 Predictions", "🔄 Nonlinear & Kernels", "⚡ Kernel Deep Dive", "🌟 Kernel Gallery", "📈 Probability Estimation"])

# Tab 0: Vector Fundamentals - START HERE!
with tabs[0]:
    st.header("🎓 Start Here: Understanding Vectors & Geometry")
    st.markdown("**Foundation concepts you need before diving into SVM!**")

    st.info("ℹ️ **Students: Start here!** This tab explains the fundamental vector concepts that make SVM work. Understanding these will make everything else crystal clear!")

    # Part 1: What is a vector?
    st.markdown("### Part 1: What is a Vector?")

    col_v1, col_v2 = st.columns([1, 1])

    with col_v1:
        st.markdown("""
        A **vector** is an arrow with:
        - **Direction**: Where it points
        - **Magnitude (length)**: How long it is

        Example: **w = [3, 4]**
        - Starts at origin (0, 0)
        - Ends at point (3, 4)
        """)

        with st.expander("❓ Help: Why do we use vectors in SVM?"):
            st.markdown("""
            **Vectors are perfect for SVM because:**
            - They represent **directions** (which way the hyperplane faces)
            - They help us compute **distances** (margins)
            - They simplify **geometric calculations**

            In SVM, the weight vector **w** tells us:
            - Which direction separates the classes
            - How to compute distances from points to the hyperplane
            """)

        # Interactive vector
        st.markdown("### 🎮 Create Your Vector")
        v1 = st.slider("w₁ component", -5.0, 5.0, 3.0, 0.5, key="v1_intro")
        v2 = st.slider("w₂ component", -5.0, 5.0, 4.0, 0.5, key="v2_intro")

        vec = np.array([v1, v2])
        magnitude = np.linalg.norm(vec)

        st.markdown(f"""
        <div class='example-box'>
        <b>Your vector: w = [{v1}, {v2}]</b><br><br>

        <b>Magnitude (length):</b><br>
        ||w|| = √(w₁² + w₂²)<br>
        ||w|| = √({v1}² + {v2}²)<br>
        ||w|| = √({v1**2:.2f} + {v2**2:.2f})<br>
        ||w|| = <b>{magnitude:.3f}</b>
        </div>
        """, unsafe_allow_html=True)

    with col_v2:
        # Visualize vector
        fig_vec = go.Figure()

        # Vector arrow
        fig_vec.add_trace(go.Scatter(
            x=[0, v1], y=[0, v2],
            mode='lines+markers',
            line=dict(color='#667eea', width=4),
            marker=dict(size=[10, 15], color=['#667eea', '#764ba2']),
            name=f'w = [{v1}, {v2}]',
            showlegend=True
        ))

        # Arrow annotation
        fig_vec.add_annotation(
            x=v1, y=v2,
            ax=0, ay=0,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=3,
            arrowsize=2,
            arrowwidth=3,
            arrowcolor='#667eea'
        )

        # Magnitude arc
        if magnitude > 0:
            theta = np.linspace(0, np.arctan2(v2, v1), 20)
            arc_r = magnitude * 0.3
            fig_vec.add_trace(go.Scatter(
                x=arc_r * np.cos(theta),
                y=arc_r * np.sin(theta),
                mode='lines',
                line=dict(color='orange', width=2, dash='dot'),
                name='Direction',
                showlegend=False
            ))

        # Length line
        fig_vec.add_trace(go.Scatter(
            x=[v1/2], y=[v2/2],
            mode='text',
            text=[f'||w|| = {magnitude:.2f}'],
            textfont=dict(size=14, color='purple'),
            showlegend=False
        ))

        fig_vec.update_layout(
            title=f"Vector w with magnitude {magnitude:.2f}",
            xaxis_title="w₁",
            yaxis_title="w₂",
            height=400,
            plot_bgcolor='#f8f9fa',
            xaxis=dict(range=[-6, 6], gridcolor='lightgray', zeroline=True, zerolinecolor='black', zerolinewidth=2),
            yaxis=dict(range=[-6, 6], gridcolor='lightgray', zeroline=True, zerolinecolor='black', zerolinewidth=2),
            showlegend=True
        )

        st.plotly_chart(fig_vec, use_container_width=True)

    st.markdown("---")

    # Part 2: Normalization - Direction vs Distance
    st.markdown("### Part 2: Direction vs Distance - Why Divide by ||w||?")

    col_n1, col_n2 = st.columns([1, 1])

    with col_n1:
        st.markdown("""
        ### 🧭 Getting the Direction (Unit Vector)

        When we divide w by its length ||w||, we get a **unit vector**:
        """)

        st.latex(r"\hat{w} = \frac{w}{||w||}")

        st.markdown("""
        - **Length = 1** (unit vector)
        - **Same direction** as w
        - **Just the direction**, no magnitude
        """)

        if magnitude > 0:
            unit_vec = vec / magnitude
            st.markdown(f"""
            <div class='formula-box'>
            <b>Your unit vector:</b><br>
            ŵ = [{v1}, {v2}] / {magnitude:.3f}<br>
            ŵ = [{unit_vec[0]:.3f}, {unit_vec[1]:.3f}]<br><br>

            Length: ||ŵ|| = {np.linalg.norm(unit_vec):.3f} ✓ (always 1!)
            </div>
            """, unsafe_allow_html=True)

        with st.expander("❓ Help: When do we use unit vectors?"):
            st.markdown("""
            **Unit vectors are used when we only care about DIRECTION:**

            Examples:
            - **Compass direction**: North, South (direction only)
            - **Normal to a surface**: Which way is "perpendicular"
            - **SVM hyperplane orientation**: Which way does it face

            We normalize w to get the direction perpendicular to the hyperplane!
            """)

    with col_n2:
        st.markdown("""
        ### 📏 Getting Distance

        To find the **distance** from a point **x** to the hyperplane, we use:
        """)

        st.latex(r"\text{distance} = \frac{|w^T x + b|}{||w||}")

        st.markdown("""
        - **Numerator**: Score (signed distance × ||w||)
        - **Denominator**: ||w|| (converts to actual distance)
        - **Division by ||w||**: Normalizes to get true geometric distance
        """)

        with st.expander("❓ Help: Why divide by ||w|| for distance?"):
            st.markdown("""
            **Think of it like this:**

            The value **wᵀx + b** tells us:
            - Which side of the hyperplane (sign)
            - How far (but scaled by ||w||)

            To get the **actual geometric distance**, we must divide by ||w||!

            **Example:**
            - If w = [3, 4], then ||w|| = 5
            - If wᵀx + b = 10
            - Distance = 10 / 5 = **2 units**

            If we doubled w to [6, 8]:
            - ||w|| = 10 (doubled!)
            - wᵀx + b = 20 (doubled!)
            - Distance = 20 / 10 = **2 units** (same!)

            **The division by ||w|| makes distance independent of w's scale!**
            """)

    st.markdown("---")

    # Part 3: Hyperplane and Margins
    st.markdown("### Part 3: Decision Hyperplane & Margins")

    st.markdown("""
    In SVM, we have **three parallel hyperplanes**:
    """)

    col_h1, col_h2 = st.columns([1, 1])

    with col_h1:
        st.markdown("""
        ### The Three Hyperplanes:

        **Decision Boundary**: wᵀx + b = 0
        - The middle line where classification happens
        - If wᵀx + b > 0 → Class +1
        - If wᵀx + b < 0 → Class -1

        **Upper Margin**: wᵀx + b = +1
        - Parallel to decision boundary
        - Distance = 1/||w|| above decision boundary
        - Class +1 points should be above this

        **Lower Margin**: wᵀx + b = -1
        - Parallel to decision boundary
        - Distance = 1/||w|| below decision boundary
        - Class -1 points should be below this
        """)

        st.markdown(f"""
        <div class='example-box'>
        <b>With your w = [{v1}, {v2}]:</b><br><br>

        ||w|| = {magnitude:.3f}<br>
        Distance from center to each margin = 1/{magnitude:.3f} = {1/magnitude if magnitude > 0 else 0:.3f}<br>
        <b>Total margin width = 2/||w|| = {2/magnitude if magnitude > 0 else 0:.3f}</b>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("❓ Help: Why are margins at +1 and -1?"):
            st.markdown("""
            **hmmm!** This seems arbitrary, but it's a smart choice:

            **Why +1 and -1?**
            - We can always scale w and b to make the margins ±1
            - This is called the "canonical form"
            - It simplifies the math!

            **What if we used ±2 instead?**
            - We could! But then we'd scale w and b differently
            - The geometry would be the same
            - The math would be messier

            **The key insight:**
            - Margin width = 2/||w||
            - To maximize margin, we minimize ||w||!
            - The ±1 convention makes this optimization clean
            """)

    with col_h2:
        # Visualize hyperplanes
        fig_hyper = go.Figure()

        # Sample points
        np.random.seed(42)
        X_pos = np.random.randn(8, 2) * 0.5 + np.array([2, 2])
        X_neg = np.random.randn(8, 2) * 0.5 + np.array([-2, -2])

        fig_hyper.add_trace(go.Scatter(
            x=X_pos[:, 0], y=X_pos[:, 1],
            mode='markers',
            marker=dict(size=10, color='#667eea'),
            name='Class +1'
        ))

        fig_hyper.add_trace(go.Scatter(
            x=X_neg[:, 0], y=X_neg[:, 1],
            mode='markers',
            marker=dict(size=10, color='#f56565'),
            name='Class -1'
        ))

        if magnitude > 0 and abs(v2) > 0.01:
            x_line = np.linspace(-4, 4, 100)

            # Decision boundary (b=0 for visualization)
            y_decision = -(v1 * x_line) / v2

            fig_hyper.add_trace(go.Scatter(
                x=x_line, y=y_decision,
                mode='lines',
                line=dict(color='green', width=4),
                name='Decision: f(x)=0'
            ))

            # Upper margin
            y_upper = -(v1 * x_line - 1) / v2
            fig_hyper.add_trace(go.Scatter(
                x=x_line, y=y_upper,
                mode='lines',
                line=dict(color='green', width=2, dash='dash'),
                name='Upper: f(x)=+1'
            ))

            # Lower margin
            y_lower = -(v1 * x_line + 1) / v2
            fig_hyper.add_trace(go.Scatter(
                x=x_line, y=y_lower,
                mode='lines',
                line=dict(color='green', width=2, dash='dash'),
                name='Lower: f(x)=-1'
            ))

            # Show w vector (perpendicular to hyperplane)
            fig_hyper.add_trace(go.Scatter(
                x=[0, v1*0.5], y=[0, v2*0.5],
                mode='lines+markers',
                line=dict(color='purple', width=3),
                marker=dict(size=10),
                name='w (perpendicular!)'
            ))

        fig_hyper.update_layout(
            title="Three Parallel Hyperplanes",
            xaxis_title="x₁",
            yaxis_title="x₂",
            height=450,
            plot_bgcolor='#f8f9fa',
            xaxis=dict(range=[-4, 4], gridcolor='lightgray', zeroline=True),
            yaxis=dict(range=[-4, 4], gridcolor='lightgray', zeroline=True),
            showlegend=True
        )

        st.plotly_chart(fig_hyper, use_container_width=True)

    st.markdown("---")

    # Part 4: Why w is perpendicular
    st.markdown("### Part 4: Understanding Perpendicularity")

    col_p1, col_p2 = st.columns([1, 1])

    with col_p1:
        st.markdown("""
        ### 🔍 The Math Behind It

        The hyperplane is defined by all points x such that:
        """)

        st.latex(r"w^T x + b = 0")

        st.markdown("""
        **Key insight:** If two points x₁ and x₂ are both on the hyperplane, then:
        - wᵀx₁ + b = 0
        - wᵀx₂ + b = 0

        Subtracting: wᵀ(x₁ - x₂) = 0

        **This means w is perpendicular to (x₁ - x₂)!**

        Since (x₁ - x₂) is a vector along the hyperplane, w must be perpendicular to the hyperplane!
        """)

        with st.expander("❓ Help: Prove that wᵀ(x₁-x₂) = 0 means perpendicular"):
            st.markdown("""
            **Dot Product = 0 means Perpendicular!**

            Remember from linear algebra:
            - **Dot product**: wᵀv = ||w|| ||v|| cos(θ)
            - If wᵀv = 0, then cos(θ) = 0
            - Therefore θ = 90° → **perpendicular!**

            **Visual intuition:**
            - Think of the hyperplane as a flat surface
            - Any vector along the surface has wᵀv = 0
            - w points "straight out" from the surface
            - This is the **normal vector**!
            """)

    with col_p2:
        st.markdown("""
        ### Why This Matters for SVM

        **Because w is perpendicular:**

        1. **Distance formula works**: Distance from point to hyperplane is:
        """)

        st.latex(r"\frac{|w^T x + b|}{||w||}")

        st.markdown("""
        2. **Margin is well-defined**: The gap between hyperplanes is:
        """)

        st.latex(r"\frac{2}{||w||}")

        st.markdown("""
        3. **Optimization makes sense**: To maximize margin, minimize ||w||!

        **Interactive Check:**
        """)

        # Check perpendicularity
        if magnitude > 0 and abs(v2) > 0.01:
            # Two points on the hyperplane
            x1_hyper = np.array([1.0, -(v1 * 1.0) / v2])
            x2_hyper = np.array([2.0, -(v1 * 2.0) / v2])
            along_hyperplane = x2_hyper - x1_hyper

            dot_product = vec @ along_hyperplane

            st.markdown(f"""
            <div class='example-box'>
            <b>Perpendicularity Check:</b><br><br>

            Point on hyperplane: x₁ = [{x1_hyper[0]:.2f}, {x1_hyper[1]:.2f}]<br>
            Another point: x₂ = [{x2_hyper[0]:.2f}, {x2_hyper[1]:.2f}]<br><br>

            Vector along hyperplane:<br>
            v = x₂ - x₁ = [{along_hyperplane[0]:.2f}, {along_hyperplane[1]:.2f}]<br><br>

            Dot product w<sup>T</sup>v:<br>
            = {v1}×{along_hyperplane[0]:.2f} + {v2}×{along_hyperplane[1]:.2f}<br>
            = <b>{dot_product:.6f}</b> ≈ 0 ✓<br><br>

            <b>w is perpendicular to the hyperplane!</b>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Part 5: Why minimize ||w||
    st.markdown("### Part 5: The Optimization Goal")

    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        st.markdown("""
        ### The Goal: Maximum Margin

        **Margin width** = Distance between the two margin hyperplanes
        """)

        st.latex(r"\text{Margin} = \frac{2}{||w||}")

        st.markdown("""
        **To maximize margin:**
        - Make the denominator ||w|| **small**
        - Therefore: **minimize ||w||** (or equivalently ||w||²)

        **Subject to constraints:**
        - All Class +1 points: wᵀx + b ≥ +1
        - All Class -1 points: wᵀx + b ≤ -1

        Or combined: yᵢ(wᵀxᵢ + b) ≥ 1 for all i
        """)

        with st.expander("❓ Help: Why maximize the margin?"):
            st.markdown("""
            **Why do we want a large margin?**

            1. **Better generalization**:
               - Large margin = more "breathing room"
               - New test points are less likely to be misclassified
               - More robust to noise

            2. **Unique solution**:
               - Maximum margin gives ONE unique hyperplane
               - Without this, infinite valid hyperplanes exist!

            3. **Statistical learning theory**:
               - Larger margin → better generalization bounds
               - Proven mathematically!

            **Intuition:** Would you rather walk on a wide path or a tightrope? Same idea - wider margin is safer!
            """)

        # Interactive margin demonstration
        st.markdown("### 🎮 Try Different ||w|| Values")

        w_norm_demo = st.slider("Set ||w|| value", 0.5, 5.0, 2.0, 0.1, key="w_norm_demo")
        margin_demo = 2 / w_norm_demo

        st.markdown(f"""
        <div class='formula-box'>
        <b>If ||w|| = {w_norm_demo:.2f}:</b><br><br>

        Margin = 2/||w|| = 2/{w_norm_demo:.2f} = <b>{margin_demo:.3f}</b><br><br>

        {'→ Larger ||w|| = Smaller margin (bad!)' if w_norm_demo > 2.5 else ''}
        {'→ Smaller ||w|| = Larger margin (good!)' if w_norm_demo < 1.5 else ''}
        {'⚖️ Medium ||w|| → Medium margin' if 1.5 <= w_norm_demo <= 2.5 else ''}
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        # Visualize margin changes
        fig_margin_demo = go.Figure()

        # For visualization, use w = [1, 1] and scale it
        w_vis = np.array([1.0, 1.0]) * w_norm_demo / np.sqrt(2)

        x_line = np.linspace(-3, 3, 100)
        y_decision = -x_line  # Decision boundary

        # Margin offset
        margin_offset = margin_demo / 2 * np.sqrt(2)

        # Three hyperplanes
        fig_margin_demo.add_trace(go.Scatter(
            x=x_line, y=y_decision,
            mode='lines',
            line=dict(color='green', width=4),
            name='Decision (f=0)'
        ))

        fig_margin_demo.add_trace(go.Scatter(
            x=x_line, y=y_decision + margin_offset,
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name=f'Upper (f=+1)'
        ))

        fig_margin_demo.add_trace(go.Scatter(
            x=x_line, y=y_decision - margin_offset,
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name=f'Lower (f=-1)'
        ))

        # Margin width indicator
        fig_margin_demo.add_trace(go.Scatter(
            x=[0, 0],
            y=[y_decision[50] - margin_offset, y_decision[50] + margin_offset],
            mode='lines+markers',
            line=dict(color='orange', width=3),
            marker=dict(size=10),
            name=f'Margin = {margin_demo:.2f}'
        ))

        fig_margin_demo.update_layout(
            title=f"Margin Width = {margin_demo:.2f} (||w|| = {w_norm_demo:.2f})",
            xaxis_title="x₁",
            yaxis_title="x₂",
            height=400,
            plot_bgcolor='#f8f9fa',
            xaxis=dict(range=[-3, 3], gridcolor='lightgray', zeroline=True),
            yaxis=dict(range=[-3, 3], gridcolor='lightgray', zeroline=True)
        )

        st.plotly_chart(fig_margin_demo, use_container_width=True)

        if w_norm_demo < 1.0:
            st.success("🎉 Small ||w|| = Large margin! SVM wants to minimize ||w||!")
        elif w_norm_demo > 3.0:
            st.warning("⚠️ Large ||w|| = Small margin! SVM avoids this!")

    st.markdown("---")

    # Summary
    st.markdown("### Summary")

    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        st.markdown("""
        <div class='example-box'>
        <b>Vectors</b><br><br>
        • Vector = direction + magnitude<br>
        • ||w|| = length of w<br>
        • w/||w|| = direction only<br>
        • Used for geometry
        </div>
        """, unsafe_allow_html=True)

    with col_s2:
        st.markdown("""
        <div class='example-box'>
        <b>Hyperplanes</b><br><br>
        • Decision: w<sup>T</sup>x + b = 0<br>
        • Margins: w<sup>T</sup>x + b = ±1<br>
        • w is perpendicular!<br>
        • Margin = 2/||w||
        </div>
        """, unsafe_allow_html=True)

    with col_s3:
        st.markdown("""
        <div class='example-box'>
        <b>Optimization</b><br><br>
        • Goal: Maximize margin<br>
        • How: Minimize ||w||<br>
        • Subject to: y<sub>i</sub>(w<sup>T</sup>x<sub>i</sub>+b)≥1<br>
        • Gets best separator!
        </div>
        """, unsafe_allow_html=True)

    st.success("**Now you're ready!** Proceed to the next tabs to see SVM in action with these concepts.")

# Tab 1: SVM Basics
with tabs[1]:
    st.header("Understanding SVM Fundamentals")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        ### What is SVM? 🤔

        Support Vector Machine (SVM) is a powerful supervised learning algorithm that finds the **optimal hyperplane**
        to separate different classes in your data.

        ### Key Concepts:

        **1. Hyperplane** 📏
        - A decision boundary that separates classes
        - In 2D: A line
        - In 3D: A plane
        - In higher dimensions: A hyperplane

        **2. Support Vectors** ⭐
        - Data points closest to the hyperplane
        - These are the critical points that define the boundary
        - Removing other points won't change the hyperplane!

        **3. Margin** 📐
        - Distance between the hyperplane and the nearest data points
        - SVM tries to **maximize** this margin
        - Larger margin = better generalization

        **4. Decision Function**
        """)

        st.latex(r"f(x) = w^T x + b")

        st.markdown("""
        - **w**: weight vector (perpendicular to hyperplane)
        - **b**: bias term
        - Sign of f(x) determines the class
        """)

    with col2:
        # Create a simple visualization
        np.random.seed(42)
        X_class1 = np.random.randn(20, 2) + np.array([2, 2])
        X_class2 = np.random.randn(20, 2) + np.array([-2, -2])
        X_demo = np.vstack([X_class1, X_class2])
        y_demo = np.array([1]*20 + [-1]*20)

        clf_demo = SVC(kernel='linear', C=1000)
        clf_demo.fit(X_demo, y_demo)

        fig = go.Figure()

        # Plot data points
        fig.add_trace(go.Scatter(
            x=X_class1[:, 0], y=X_class1[:, 1],
            mode='markers',
            marker=dict(size=12, color='#667eea', line=dict(width=2, color='#5a67d8')),
            name='Class 1'
        ))

        fig.add_trace(go.Scatter(
            x=X_class2[:, 0], y=X_class2[:, 1],
            mode='markers',
            marker=dict(size=12, color='#f56565', line=dict(width=2, color='#e53e3e')),
            name='Class -1'
        ))

        # Plot support vectors
        sv = clf_demo.support_vectors_
        fig.add_trace(go.Scatter(
            x=sv[:, 0], y=sv[:, 1],
            mode='markers',
            marker=dict(size=18, color='#ffd700', symbol='star',
                       line=dict(width=3, color='#ff8c00')),
            name='Support Vectors ⭐'
        ))

        # Decision boundary
        w = clf_demo.coef_[0]
        b = clf_demo.intercept_[0]
        x_line = np.linspace(X_demo[:, 0].min()-1, X_demo[:, 0].max()+1, 100)
        y_line = -(w[0] * x_line + b) / w[1]

        fig.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode='lines',
            line=dict(color='#48bb78', width=4),
            name='Decision Boundary'
        ))

        # Margins
        margin = 1 / np.sqrt(np.sum(w ** 2))
        y_margin_up = y_line + margin * np.sqrt(1 + (w[0]/w[1])**2)
        y_margin_down = y_line - margin * np.sqrt(1 + (w[0]/w[1])**2)

        fig.add_trace(go.Scatter(
            x=x_line, y=y_margin_up,
            mode='lines',
            line=dict(color='#48bb78', width=2, dash='dash'),
            name='Margin',
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=x_line, y=y_margin_down,
            mode='lines',
            line=dict(color='#48bb78', width=2, dash='dash'),
            name='Margin'
        ))

        fig.update_layout(
            title="SVM: Maximum Margin Classifier",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            height=600,
            showlegend=True,
            plot_bgcolor='#f8f9fa',
            hovermode='closest',
            font=dict(size=12)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class='example-box'>
        <b>📊 In this example:</b><br>
        • Green solid line: Decision boundary (hyperplane)<br>
        • Green dashed lines: Margins<br>
        • Gold stars ⭐: Support vectors (critical points)<br>
        • Margin width: <b>{2*margin:.3f}</b> units<br>
        • Number of support vectors: <b>{len(sv)}</b>
        </div>
        """, unsafe_allow_html=True)

# Tab 2: Interactive SVM
with tabs[2]:
    st.header("🎨 Interactive SVM Playground")
    st.markdown("**Adjust the controls below and see the SVM update in real-time!**")

    st.info("""
    💡 **Pro Tip**: To see the C parameter effect clearly:
    1. Use **"Overlapping"** dataset (default)
    2. Change C from **0.01** (very soft) to **100** (very strict)
    3. Watch **Margin Violations**, **Test Acc**, and the **timestamp** change!
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🎛️ Controls")

        n_samples = st.slider(
            "📊 Number of samples per class",
            min_value=20,
            max_value=200,
            value=50,
            step=10,
            key="interactive_n_samples",
            help="More samples = more data points to learn from"
        )

        dataset_type = st.selectbox(
            "📈 Dataset type",
            ["Linearly Separable", "With Noise", "Overlapping"],
            index=2,  # Default to "Overlapping" to show C effect better
            key="interactive_dataset_type",
            help="Choose how difficult the classification problem is"
        )

        # Use select_slider with logarithmic scale for better C control
        C_options = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
        C_param = st.select_slider(
            "⚖️ C (Regularization)",
            options=C_options,
            value=1.0,
            key="interactive_C_param",
            help="Smaller C = wider margin, more errors allowed | Larger C = narrower margin, fewer errors"
        )

        # Show current C interpretation with exact value
        if C_param <= 0.1:
            st.info(f"🔵 **C = {C_param}**: Very soft margin - prioritizes large margin, many violations allowed")
        elif C_param <= 1.0:
            st.info(f"🟢 **C = {C_param}**: Balanced - moderate margin with some violations")
        elif C_param <= 10.0:
            st.warning(f"🟡 **C = {C_param}**: Stricter - smaller margin, fewer violations")
        else:
            st.error(f"🔴 **C = {C_param}**: Very strict - minimal violations, smallest margin")

        st.markdown("---")
        st.subheader("👁️ Visualization Options")

        show_vectors = st.checkbox(
            "📐 Show weight vector w",
            value=True,
            key="interactive_show_vectors"
        )
        show_margins = st.checkbox(
            "📏 Show margins",
            value=True,
            key="interactive_show_margins"
        )
        show_support = st.checkbox(
            "⭐ Highlight support vectors",
            value=True,
            key="interactive_show_support"
        )

        st.markdown("---")

        if st.button("🎲 Generate New Random Data", key="interactive_generate_data", use_container_width=True):
            st.session_state.random_seed = np.random.randint(0, 1000)
            st.rerun()

        # Show current settings with update indicator
        st.markdown("---")

        # Add timestamp to prove updates are happening
        import datetime
        update_time = datetime.datetime.now().strftime("%H:%M:%S")

        st.markdown("**📋 Current Settings:**")
        st.markdown(f"""
        - Samples per class: **{n_samples}** (Total: {n_samples * 2})
        - Dataset: **{dataset_type}**
        - C parameter: **{C_param}**
        - Random seed: **{st.session_state.get('random_seed', 42)}**
        - 🔄 Last update: **{update_time}**
        """)

        st.info("ℹ️ **Watch the timestamp** - it updates every time you change a slider, proving the model is retraining!")

    with col2:
        # Generate data based on selection
        seed = st.session_state.get('random_seed', 42)
        np.random.seed(seed)

        if dataset_type == "Linearly Separable":
            # Very easy: well separated, small variance
            X_class1 = np.random.randn(n_samples, 2) * 0.5 + np.array([2, 2])
            X_class2 = np.random.randn(n_samples, 2) * 0.5 + np.array([-2, -2])
        elif dataset_type == "With Noise":
            # Medium: some noise, still mostly separable
            X_class1 = np.random.randn(n_samples, 2) * 1.0 + np.array([2, 2])
            X_class2 = np.random.randn(n_samples, 2) * 1.0 + np.array([-2, -2])
        else:  # Overlapping
            # Hard: significant overlap, C parameter effect will be very visible!
            X_class1 = np.random.randn(n_samples, 2) * 2.0 + np.array([1, 1])
            X_class2 = np.random.randn(n_samples, 2) * 2.0 + np.array([-1, -1])

        X_interactive = np.vstack([X_class1, X_class2])
        y_interactive = np.array([1]*n_samples + [-1]*n_samples)

        # Generate small test set (20% of data) for generalization accuracy
        n_test = max(5, n_samples // 5)  # At least 5 samples per class for test
        if dataset_type == "Linearly Separable":
            X_test1 = np.random.randn(n_test, 2) * 0.5 + np.array([2, 2])
            X_test2 = np.random.randn(n_test, 2) * 0.5 + np.array([-2, -2])
        elif dataset_type == "With Noise":
            X_test1 = np.random.randn(n_test, 2) * 1.0 + np.array([2, 2])
            X_test2 = np.random.randn(n_test, 2) * 1.0 + np.array([-2, -2])
        else:  # Overlapping
            X_test1 = np.random.randn(n_test, 2) * 2.0 + np.array([1, 1])
            X_test2 = np.random.randn(n_test, 2) * 2.0 + np.array([-1, -1])

        X_test = np.vstack([X_test1, X_test2])
        y_test = np.array([1]*n_test + [-1]*n_test)

        # Train SVM
        clf_interactive = SVC(kernel='linear', C=C_param)
        clf_interactive.fit(X_interactive, y_interactive)

        # Create mesh for decision boundary
        x_min, x_max = X_interactive[:, 0].min() - 1, X_interactive[:, 0].max() + 1
        y_min, y_max = X_interactive[:, 1].min() - 1, X_interactive[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))

        Z = clf_interactive.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig_interactive = go.Figure()

        # Decision boundary contour
        fig_interactive.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, 200),
            y=np.linspace(y_min, y_max, 200),
            z=Z,
            colorscale=[[0, 'rgba(245,101,101,0.3)'], [0.5, 'rgba(255,255,255,0)'],
                       [1, 'rgba(102,126,234,0.3)']],
            showscale=False,
            contours=dict(
                start=-2,
                end=2,
                size=0.5,
            ),
            name='Decision Function',
            hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>f(x): %{z:.2f}<extra></extra>'
        ))

        # Training data points
        fig_interactive.add_trace(go.Scatter(
            x=X_class1[:, 0], y=X_class1[:, 1],
            mode='markers',
            marker=dict(size=10, color='#667eea', line=dict(width=2, color='#5a67d8')),
            name='Class +1 (train)'
        ))

        fig_interactive.add_trace(go.Scatter(
            x=X_class2[:, 0], y=X_class2[:, 1],
            mode='markers',
            marker=dict(size=10, color='#f56565', line=dict(width=2, color='#e53e3e')),
            name='Class -1 (train)'
        ))

        # Test data points (smaller, semi-transparent)
        fig_interactive.add_trace(go.Scatter(
            x=X_test1[:, 0], y=X_test1[:, 1],
            mode='markers',
            marker=dict(size=7, color='#667eea', line=dict(width=1, color='#5a67d8'),
                       opacity=0.6),
            name='Class +1 (test)'
        ))

        fig_interactive.add_trace(go.Scatter(
            x=X_test2[:, 0], y=X_test2[:, 1],
            mode='markers',
            marker=dict(size=7, color='#f56565', line=dict(width=1, color='#e53e3e'),
                       opacity=0.6),
            name='Class -1 (test)'
        ))

        # Weight vector and bias (needed for support vector categorization)
        w = clf_interactive.coef_[0]
        b = clf_interactive.intercept_[0]

        # Support vectors - categorize and color-code them
        if show_support:
            sv = clf_interactive.support_vectors_
            sv_indices = clf_interactive.support_
            sv_labels = y_interactive[sv_indices]

            # Categorize support vectors
            sv_on_margin_x, sv_on_margin_y = [], []
            sv_inside_x, sv_inside_y = [], []
            sv_misclass_x, sv_misclass_y = [], []

            for i, sv_point in enumerate(sv):
                decision_value = sv_labels[i] * (w @ sv_point + b)
                if abs(decision_value - 1.0) < 0.01:  # On margin (≈1)
                    sv_on_margin_x.append(sv_point[0])
                    sv_on_margin_y.append(sv_point[1])
                elif decision_value >= 0:  # Inside margin but correct side
                    sv_inside_x.append(sv_point[0])
                    sv_inside_y.append(sv_point[1])
                else:  # Wrong side (misclassified)
                    sv_misclass_x.append(sv_point[0])
                    sv_misclass_y.append(sv_point[1])

            # Plot support vectors on margin (gold stars)
            if sv_on_margin_x:
                fig_interactive.add_trace(go.Scatter(
                    x=sv_on_margin_x, y=sv_on_margin_y,
                    mode='markers',
                    marker=dict(size=16, color='#ffd700', symbol='star',
                               line=dict(width=3, color='#ff8c00')),
                    name='SV: On Margin ⭐'
                ))

            # Plot support vectors inside margin (orange circles)
            if sv_inside_x:
                fig_interactive.add_trace(go.Scatter(
                    x=sv_inside_x, y=sv_inside_y,
                    mode='markers',
                    marker=dict(size=14, color='#ff9800', symbol='circle',
                               line=dict(width=3, color='#f57c00')),
                    name='SV: Inside Margin 🔸'
                ))

            # Plot misclassified support vectors (red X)
            if sv_misclass_x:
                fig_interactive.add_trace(go.Scatter(
                    x=sv_misclass_x, y=sv_misclass_y,
                    mode='markers',
                    marker=dict(size=16, color='#ff0000', symbol='x',
                               line=dict(width=3, color='#cc0000')),
                    name='SV: Misclassified ✗'
                ))

        if show_vectors:
            # Show w vector from origin
            fig_interactive.add_trace(go.Scatter(
                x=[0, w[0]*1.5], y=[0, w[1]*1.5],
                mode='lines+markers',
                line=dict(color='#9f7aea', width=4),
                marker=dict(size=12, symbol='arrow', angleref='previous'),
                name=f'w = [{w[0]:.2f}, {w[1]:.2f}]'
            ))

        # Decision boundary line
        x_line = np.linspace(x_min, x_max, 100)
        y_line = -(w[0] * x_line + b) / w[1]

        fig_interactive.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode='lines',
            line=dict(color='#48bb78', width=4),
            name='Decision Boundary'
        ))

        if show_margins:
            # Margin lines
            norm_w = np.sqrt(np.sum(w ** 2))
            margin = 1 / norm_w

            y_margin_up = -(w[0] * x_line + b - 1) / w[1]
            y_margin_down = -(w[0] * x_line + b + 1) / w[1]

            fig_interactive.add_trace(go.Scatter(
                x=x_line, y=y_margin_up,
                mode='lines',
                line=dict(color='#48bb78', width=2, dash='dash'),
                name='Upper Margin'
            ))

            fig_interactive.add_trace(go.Scatter(
                x=x_line, y=y_margin_down,
                mode='lines',
                line=dict(color='#48bb78', width=2, dash='dash'),
                name='Lower Margin'
            ))

        # Calculate metrics for title
        norm_w_title = np.sqrt(np.sum(w ** 2))
        margin_title = 2 / norm_w_title
        total_samples = len(X_interactive)
        test_acc = clf_interactive.score(X_test, y_test) * 100

        fig_interactive.update_layout(
            title=f"Interactive SVM | Train: {total_samples}, Test: {len(X_test)} | C = {C_param} | Test Acc = {test_acc:.1f}% | SV = {len(clf_interactive.support_vectors_)}",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            height=600,
            showlegend=True,
            plot_bgcolor='#f8f9fa'
        )

        st.plotly_chart(fig_interactive, use_container_width=True)

        # Show parameters
        norm_w = np.sqrt(np.sum(w ** 2))
        margin = 1 / norm_w

        # Calculate margin violations (points with slack variables)
        decision_values = clf_interactive.decision_function(X_interactive)
        margin_violations = np.sum(np.abs(y_interactive * decision_values) < 1)
        misclassified_train = np.sum(y_interactive * decision_values < 0)

        # Calculate both training and test accuracy
        train_acc = clf_interactive.score(X_interactive, y_interactive) * 100
        test_acc_metric = clf_interactive.score(X_test, y_test) * 100

        # Visual indicator of model state based on metrics
        if margin_violations > len(X_interactive) * 0.3:
            model_state = "🔵 Very Soft Margin (Many violations)"
            state_color = "blue"
        elif margin_violations > len(X_interactive) * 0.1:
            model_state = "🟢 Balanced (Some violations)"
            state_color = "green"
        elif margin_violations > 0:
            model_state = "🟡 Strict (Few violations)"
            state_color = "orange"
        else:
            model_state = "🔴 Very Strict (No violations - may overfit!)"
            state_color = "red"

        st.markdown(f"### 📊 Model Performance: {model_state}")

        col_a, col_b, col_c, col_d, col_e, col_f = st.columns(6)
        with col_a:
            st.metric("||w||", f"{norm_w:.3f}", help="Magnitude of weight vector")
        with col_b:
            st.metric("Margin", f"{2*margin:.3f}", help="Margin width = 2/||w||")
        with col_c:
            st.metric("Support Vectors", len(clf_interactive.support_vectors_))
        with col_d:
            # Show violations with color based on amount
            violation_pct = (margin_violations / len(X_interactive)) * 100
            st.metric("Margin Violations", f"{margin_violations} ({violation_pct:.0f}%)",
                     help="Training points inside margin (have slack ξ > 0)")
        with col_e:
            st.metric("Train Acc", f"{train_acc:.1f}%",
                     help="Accuracy on training data")
        with col_f:
            delta = test_acc_metric - train_acc
            st.metric("Test Acc", f"{test_acc_metric:.1f}%",
                     delta=f"{delta:+.1f}%",
                     help="Accuracy on unseen test data - shows generalization!")

        # Help explaining C parameter and support vectors
        with st.expander("❓ Help: Understanding Metrics and Accuracy"):
            st.markdown(f"""
            **Current Status (C = {C_param}):**
            - Training Accuracy: {train_acc:.1f}%
            - Test Accuracy: {test_acc_metric:.1f}%
            - Margin Violations: {margin_violations} out of {len(X_interactive)} training points

            **Why Training Accuracy Can Be 100% Even With Overlapping Data:**

            In soft-margin SVM, "accuracy" means the predicted label matches the true label:
            - f(x) > 0 → predict +1 → if true label is +1, it's "correct"
            - f(x) < 0 → predict -1 → if true label is -1, it's "correct"

            **BUT** a point can be:
            - ✓ Correctly classified (100% training accuracy)
            - ✗ Inside the margin (violation, ξ > 0)
            - ✗ Even slightly on wrong side (soft-margin allows this!)

            The SVM uses **slack variables (ξ)** to handle violations:
            - Point far from boundary: ξ = 0 (no violation)
            - Point inside margin: 0 < ξ < 1 (violation, but correct side)
            - Misclassified: ξ > 1 (heavy penalty from C)

            **This is why you see:**
            - Training Acc = 100% (all labels predicted correctly)
            - Margin Violations > 0 (some points have ξ > 0)
            - These are NOT contradictory!

            **Test Accuracy is the Real Metric:**
            - Shows how well the model generalizes to NEW data
            - Can be lower than training accuracy (overfitting)
            - Can be higher than training accuracy (good generalization)
            - **Watch test accuracy change with C!**

            **How C affects this:**
            - **Small C**: More violations allowed, better test accuracy (generalization)
            - **Large C**: Fewer violations, may overfit, worse test accuracy
            - **Try it**: Change C with "Overlapping" dataset and watch test accuracy!
            """)

        # Show support vector analysis
        if show_support and len(clf_interactive.support_vectors_) > 0:
            st.markdown("---")
            st.markdown("### 🔍 Support Vector Analysis")

            # Categorize support vectors
            sv = clf_interactive.support_vectors_
            sv_indices = clf_interactive.support_
            sv_labels = y_interactive[sv_indices]

            sv_on_margin = 0
            sv_inside_margin = 0
            sv_misclassified = 0

            for i, sv_point in enumerate(sv):
                decision_value = sv_labels[i] * (w @ sv_point + b)
                if abs(decision_value - 1.0) < 0.01:  # On margin (≈1)
                    sv_on_margin += 1
                elif decision_value >= 0:  # Inside margin but correct side
                    sv_inside_margin += 1
                else:  # Wrong side (misclassified)
                    sv_misclassified += 1

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("On Margin Boundary", sv_on_margin,
                         help="Support vectors exactly at distance 1/||w|| from boundary")
            with col2:
                st.metric("Inside Margin", sv_inside_margin,
                         help="Support vectors between decision boundary and margin")
            with col3:
                st.metric("Misclassified", sv_misclassified,
                         help="Support vectors on wrong side of decision boundary")

            st.info(f"ℹ️ **Current C = {C_param}**: You have {sv_inside_margin + sv_misclassified} support vectors with violations (inside margin or misclassified). Decrease C to allow more violations and increase margin width, or increase C to reduce violations but accept smaller margin.")

# Tab 3: Simple Numerical Example
with tabs[3]:
    st.header("🧮 Simple Numerical Example")
    st.markdown("**Let's work through a concrete example with real numbers!**")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        ### Step-by-Step Example

        Let's say we have a simple SVM with:
        """)

        # Interactive example
        w1_ex = st.slider("w₁ (first weight)", -5.0, 5.0, 2.0, 0.5)
        w2_ex = st.slider("w₂ (second weight)", -5.0, 5.0, 1.5, 0.5)
        b_ex = st.slider("b (bias)", -5.0, 5.0, -1.0, 0.5)

        st.markdown(f"""
        <div class='formula-box'>
        <b>Decision Function:</b><br>
        f(x) = w₁×x₁ + w₂×x₂ + b<br>
        f(x) = {w1_ex}×x₁ + {w2_ex}×x₂ + ({b_ex})
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Test a Point")

        x1_test = st.slider("Test point x₁", -5.0, 5.0, 3.0, 0.5)
        x2_test = st.slider("Test point x₂", -5.0, 5.0, 2.0, 0.5)

        # Calculate prediction
        score = w1_ex * x1_test + w2_ex * x2_test + b_ex
        prediction = "Class +1 (Blue)" if score > 0 else "Class -1 (Red)"

        st.markdown(f"""
        <div class='example-box'>
        <b>Calculation:</b><br>
        f([{x1_test}, {x2_test}]) = {w1_ex}×{x1_test} + {w2_ex}×{x2_test} + ({b_ex})<br>
        f([{x1_test}, {x2_test}]) = {w1_ex*x1_test:.2f} + {w2_ex*x2_test:.2f} + ({b_ex})<br>
        f([{x1_test}, {x2_test}]) = <b>{score:.2f}</b><br><br>

        <b>Prediction:</b> {prediction}<br>
        <b>Confidence:</b> |{score:.2f}| = {abs(score):.2f}
        </div>
        """, unsafe_allow_html=True)

        if score > 0:
            st.success(f"✓ Score is positive ({score:.2f} > 0) → Predict **Class +1**")
        else:
            st.error(f"✓ Score is negative ({score:.2f} < 0) → Predict **Class -1**")

        st.markdown("---")

        # Margin calculation
        norm_w = np.sqrt(w1_ex**2 + w2_ex**2)
        margin_val = 2 / norm_w if norm_w > 0 else 0

        st.markdown(f"""
        <div class='example-box'>
        <b>Margin Calculation:</b><br>
        ||w|| = √(w₁² + w₂²)<br>
        ||w|| = √({w1_ex}² + {w2_ex}²)<br>
        ||w|| = √({w1_ex**2:.2f} + {w2_ex**2:.2f})<br>
        ||w|| = <b>{norm_w:.3f}</b><br><br>

        Margin = 2/||w|| = 2/{norm_w:.3f} = <b>{margin_val:.3f}</b>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Visualize the example
        fig_example = go.Figure()

        # Create grid for visualization
        x_range = np.linspace(-5, 5, 100)
        y_range = np.linspace(-5, 5, 100)
        xx_ex, yy_ex = np.meshgrid(x_range, y_range)

        # Decision function values
        zz_ex = w1_ex * xx_ex + w2_ex * yy_ex + b_ex

        # Add heatmap
        fig_example.add_trace(go.Contour(
            x=x_range,
            y=y_range,
            z=zz_ex,
            colorscale='RdBu',
            zmid=0,
            showscale=False,  # Hide colorbar to avoid overlap with text
            contours=dict(
                start=-10,
                end=10,
                size=1,
            ),
            hovertemplate='x₁: %{x:.2f}<br>x₂: %{y:.2f}<br>f(x): %{z:.2f}<extra></extra>'
        ))

        # Decision boundary (f(x) = 0)
        if w2_ex != 0:
            x_line = np.linspace(-5, 5, 100)
            y_line = -(w1_ex * x_line + b_ex) / w2_ex

            fig_example.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode='lines',
                line=dict(color='yellow', width=4),
                name='Decision Boundary (f(x)=0)'
            ))

            # Margin lines
            if norm_w > 0:
                y_margin_up = -(w1_ex * x_line + b_ex - 1) / w2_ex
                y_margin_down = -(w1_ex * x_line + b_ex + 1) / w2_ex

                fig_example.add_trace(go.Scatter(
                    x=x_line, y=y_margin_up,
                    mode='lines',
                    line=dict(color='yellow', width=2, dash='dash'),
                    name='Margin (f(x)=+1)'
                ))

                fig_example.add_trace(go.Scatter(
                    x=x_line, y=y_margin_down,
                    mode='lines',
                    line=dict(color='yellow', width=2, dash='dash'),
                    name='Margin (f(x)=-1)'
                ))

        # Test point
        fig_example.add_trace(go.Scatter(
            x=[x1_test], y=[x2_test],
            mode='markers+text',
            marker=dict(size=20, color='#ffd700', symbol='star',
                       line=dict(width=3, color='black')),
            text=[f'Test Point<br>f(x)={score:.2f}'],
            textposition='top center',
            textfont=dict(size=14, color='black', family='Arial Black'),
            name='Test Point'
        ))

        # Weight vector
        fig_example.add_trace(go.Scatter(
            x=[0, w1_ex], y=[0, w2_ex],
            mode='lines+markers+text',
            line=dict(color='#9f7aea', width=4),
            marker=dict(size=12),
            text=['', f'w=[{w1_ex}, {w2_ex}]'],
            textposition='top center',
            name='Weight Vector w'
        ))

        fig_example.update_layout(
            title=f"Visual Representation: f(x) = {w1_ex}x₁ + {w2_ex}x₂ + ({b_ex})",
            xaxis_title="x₁",
            yaxis_title="x₂",
            height=600,
            showlegend=True,
            plot_bgcolor='white',
            xaxis=dict(range=[-5, 5], gridcolor='lightgray'),
            yaxis=dict(range=[-5, 5], gridcolor='lightgray')
        )

        st.plotly_chart(fig_example, use_container_width=True)

        # Additional examples
        st.markdown("### 📝 Try These Sample Points:")

        sample_points = [
            (4, 3),
            (-3, -2),
            (0, 0),
            (1, -1)
        ]

        sample_data = []
        for x1, x2 in sample_points:
            s = w1_ex * x1 + w2_ex * x2 + b_ex
            pred = "+1" if s > 0 else "-1"
            sample_data.append({
                "Point": f"[{x1}, {x2}]",
                "Calculation": f"{w1_ex}×{x1} + {w2_ex}×{x2} + ({b_ex})",
                "Score": f"{s:.2f}",
                "Class": pred
            })

        df_samples = pd.DataFrame(sample_data)
        st.dataframe(df_samples, use_container_width=True, hide_index=True)

# Tab 4: Constraints & Formulation - NEW!
with tabs[4]:
    st.header("🔬 SVM Formulation & Constraints: Step-by-Step")
    st.markdown("**Let's build the SVM optimization problem from scratch!**")

    st.markdown("""
    ### 📖 The Story of SVM

    Imagine you have data points from two classes, and you want to draw a line (hyperplane) that:
    1. **Separates** the two classes
    2. Has the **maximum margin** (distance to nearest points)
    3. **Minimizes errors** if perfect separation isn't possible

    Let's build this step by step! 👇
    """)

    # Create sample data for demonstration
    np.random.seed(42)
    X_const = np.array([
        [1, 2], [2, 3], [2, 1],  # Class +1 (blue)
        [-1, -1], [-2, -1], [-1, -2]  # Class -1 (red)
    ])
    y_const = np.array([1, 1, 1, -1, -1, -1])

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🎛️ Design Your Hyperplane")

        st.markdown("""
        A hyperplane is defined by: **wᵀx + b = 0**

        Adjust w and b to see how constraints work:
        """)

        w1_const = st.slider("w₁ (x-direction weight)", -3.0, 3.0, 1.0, 0.1, key="w1_const")
        w2_const = st.slider("w₂ (y-direction weight)", -3.0, 3.0, 1.0, 0.1, key="w2_const")
        b_const = st.slider("b (bias)", -5.0, 5.0, -0.5, 0.1, key="b_const")

        w_const = np.array([w1_const, w2_const])
        norm_w_const = np.linalg.norm(w_const)

        st.markdown(f"""
        <div class='formula-box'>
        <b>Your Hyperplane:</b><br>
        {w1_const:.2f}×x₁ + {w2_const:.2f}×x₂ + {b_const:.2f} = 0<br><br>
        ||w|| = {norm_w_const:.3f}<br>
        Margin = 2/||w|| = {2/norm_w_const if norm_w_const > 0 else 0:.3f}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📋 Understanding Constraints")

        st.markdown("""
        For SVM, each data point must satisfy:

        **y_i × (w^T x_i + b) ≥ 1**

        This means:
        - **Class +1 points**: w^T x_i + b ≥ +1 (above upper margin)
        - **Class -1 points**: w^T x_i + b ≤ -1 (below lower margin)
        - **Support vectors**: Exactly on the margin (= ±1)
        """)

        # Calculate constraints for each point
        constraints_data = []
        for i, (x, y) in enumerate(zip(X_const, y_const)):
            score = w_const @ x + b_const
            constraint_value = y * score
            satisfied = "Yes ✓" if constraint_value >= 0.99 else "No ✗"

            constraints_data.append({
                "Point": i+1,
                "Class": f"{y:+d}",
                "Score": f"{score:.2f}",
                "y×score": f"{constraint_value:.2f}",
                "≥ 1?": satisfied,
                "Slack ξ": f"{max(0, 1 - constraint_value):.2f}"
            })

        df_constraints = pd.DataFrame(constraints_data)
        st.dataframe(df_constraints, use_container_width=True, hide_index=True)

        violations = sum([1 for row in constraints_data if "✗" in row["≥ 1?"]])
        total_slack = sum([float(row["Slack ξ"]) for row in constraints_data])

        if violations == 0:
            st.success(f"🎉 All constraints satisfied! This is a valid SVM solution.")
        else:
            st.warning(f"⚠️ {violations} constraint violations. Total slack = {total_slack:.2f}")

    with col2:
        # Visualization
        fig_const = go.Figure()

        # Plot data points
        for i, (x, y) in enumerate(zip(X_const, y_const)):
            score = w_const @ x + b_const
            constraint_value = y * score
            satisfied = constraint_value >= 0.99

            color = '#667eea' if y == 1 else '#f56565'
            marker_symbol = 'circle' if satisfied else 'x'
            marker_size = 14 if satisfied else 18

            fig_const.add_trace(go.Scatter(
                x=[x[0]], y=[x[1]],
                mode='markers+text',
                marker=dict(size=marker_size, color=color, symbol=marker_symbol,
                           line=dict(width=3, color='white')),
                text=[f"P{i+1}"],
                textposition='top center',
                name=f"Point {i+1} ({'✓' if satisfied else '✗'})",
                showlegend=True
            ))

        # Decision boundary
        if w2_const != 0:
            x_line = np.linspace(-4, 4, 100)
            y_line = -(w1_const * x_line + b_const) / w2_const

            fig_const.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode='lines',
                line=dict(color='green', width=3),
                name='Decision Boundary (f=0)'
            ))

            # Margins
            if norm_w_const > 0:
                y_margin_up = -(w1_const * x_line + b_const - 1) / w2_const
                y_margin_down = -(w1_const * x_line + b_const + 1) / w2_const

                fig_const.add_trace(go.Scatter(
                    x=x_line, y=y_margin_up,
                    mode='lines',
                    line=dict(color='green', width=2, dash='dash'),
                    name='Upper Margin (f=+1)'
                ))

                fig_const.add_trace(go.Scatter(
                    x=x_line, y=y_margin_down,
                    mode='lines',
                    line=dict(color='green', width=2, dash='dash'),
                    name='Lower Margin (f=-1)'
                ))

        # Weight vector
        fig_const.add_trace(go.Scatter(
            x=[0, w1_const], y=[0, w2_const],
            mode='lines+markers',
            line=dict(color='purple', width=3),
            marker=dict(size=10),
            name='Weight vector w'
        ))

        fig_const.update_layout(
            title="Constraint Visualization: Are all points in the right region?",
            xaxis_title="x₁",
            yaxis_title="x₂",
            height=500,
            plot_bgcolor='#f8f9fa',
            xaxis=dict(range=[-4, 4], gridcolor='lightgray', zeroline=True, zerolinecolor='black'),
            yaxis=dict(range=[-4, 4], gridcolor='lightgray', zeroline=True, zerolinecolor='black'),
            showlegend=True
        )

        st.plotly_chart(fig_const, use_container_width=True)

    st.markdown("---")

    # Optimization objective
    st.subheader("The Complete Optimization Problem")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        ### What SVM Minimizes:
        """)

        st.latex(r"""
        \min_{w,b} \quad \frac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \xi_i
        """)

        st.markdown("""
        **Two competing goals:**
        1. **Minimize ||w||²**: Make margin large (Margin = 2/||w||)
        2. **Minimize Σξᵢ**: Reduce constraint violations

        **C** controls the trade-off!
        """)

        current_objective = 0.5 * (norm_w_const**2)
        slack_penalty = total_slack

        st.markdown(f"""
        <div class='example-box'>
        <b>Current Solution:</b><br>
        • ||w||² = {norm_w_const**2:.3f}<br>
        • Total slack Σξᵢ = {total_slack:.3f}<br>
        • Margin = {2/norm_w_const if norm_w_const > 0 else 0:.3f}<br><br>

        <b>Objective value (C=1):</b><br>
        0.5×{norm_w_const**2:.3f} + 1×{total_slack:.3f} = <b>{current_objective + slack_penalty:.3f}</b>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        ### Subject to Constraints:
        """)

        st.latex(r"""
        \begin{aligned}
        &y_i(w^T x_i + b) \geq 1 - \xi_i \quad \forall i \\
        &\xi_i \geq 0 \quad \forall i
        \end{aligned}
        """)

        st.markdown("""
        **What this means:**

        1. **Hard constraint (ξᵢ = 0)**: Point exactly satisfies yᵢ(wᵀxᵢ + b) ≥ 1
        2. **Soft constraint (ξᵢ > 0)**: Point violates the margin
           - ξᵢ = amount of violation
           - We pay penalty C×ξᵢ in the objective

        **The slack variable ξᵢ allows flexibility!**
        - Without it: No solution if data isn't perfectly separable
        - With it: We can handle overlapping classes
        """)

    st.markdown("---")

    # Interactive optimization demonstration
    st.subheader("🔄 Try to Optimize!")

    col_opt1, col_opt2 = st.columns([1, 1])

    with col_opt1:
        st.markdown("""
        **Your Challenge:** Adjust w and b above to:
        1. Satisfy all constraints (all points marked ✓)
        2. 📏 Maximize the margin (make it as wide as possible)
        3. 📉 Minimize the objective function value

        **Tips:**
        - The weight vector w should point from red class toward blue class
        - Try w₁=1, w₂=1, b=-0.5 as a starting point
        - Look at the constraint table to see which points violate constraints
        - Green dashed lines show the margin - make it wide!
        """)

    with col_opt2:
        if violations == 0 and norm_w_const > 0:
            quality_score = 2 / norm_w_const  # margin width
            st.success(f"✓ Valid solution! Margin width = {quality_score:.3f}")

            #if quality_score > 1.0:
                #st.balloons()
                #st.success("🎉 Excellent! You found a solution with wide margin!")
        #else:
            st.info("Keep adjusting! Try to satisfy all constraints first.")

        # Show optimal solution hint
        if st.button("💡 Show Optimal Solution"):
            clf_opt_const = SVC(kernel='linear', C=1000)
            clf_opt_const.fit(X_const, y_const)
            w_opt = clf_opt_const.coef_[0]
            b_opt = clf_opt_const.intercept_[0]

            st.markdown(f"""
            <div class='example-box'>
            <b>Optimal Solution (found by SVM):</b><br>
            • w = [{w_opt[0]:.2f}, {w_opt[1]:.2f}]<br>
            • b = {b_opt:.2f}<br>
            • Margin = {2/np.linalg.norm(w_opt):.3f}<br>
            • All constraints satisfied ✓
            </div>
            """, unsafe_allow_html=True)

# Tab 5: Optimization (renumbered)
with tabs[5]:
    st.header("⚙️ What is SVM Optimizing?")

    st.markdown("""
    ### The SVM Optimization Problem

    SVM solves the following optimization problem:
    """)

    st.latex(r"""
    \begin{aligned}
    \text{minimize} \quad & \frac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \xi_i \\
    \text{subject to} \quad & y_i(w^T x_i + b) \geq 1 - \xi_i \\
    & \xi_i \geq 0
    \end{aligned}
    """)

    st.markdown("""
    <div class='example-box'>
    <b>What does this mean?</b><br>
    • <b>||w||²</b>: Minimize to maximize the margin (margin = 2/||w||)<br>
    • <b>C</b>: Trade-off between large margin vs few errors<br>
    • <b>ξᵢ</b>: Slack variables (allow some misclassifications)<br>
    • <b>yᵢ(wᵀxᵢ + b) ≥ 1</b>: Points must be on correct side of margin
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚖️ Effect of C Parameter")

        C_demo = st.select_slider(
            "Select C value",
            options=[0.01, 0.1, 1.0, 10.0, 100.0],
            value=1.0
        )

        st.markdown(f"""
        <div class='example-box'>
        <b>C = {C_demo}</b><br><br>

        <b>Small C (e.g., 0.01):</b><br>
        • Prioritize large margin<br>
        • Allow more misclassifications<br>
        • More regularization<br>
        • Better for noisy data<br><br>

        <b>Large C (e.g., 100):</b><br>
        • Prioritize correct classification<br>
        • Smaller margin acceptable<br>
        • Less regularization<br>
        • Risk of overfitting
        </div>
        """, unsafe_allow_html=True)

    # Generate overlapping data for demonstration
    np.random.seed(42)
    X_opt = np.vstack([
        np.random.randn(30, 2) * 1.2 + [2, 2],
        np.random.randn(30, 2) * 1.2 + [-2, -2]
    ])
    y_opt = np.array([1]*30 + [-1]*30)

    # Train SVM with selected C value
    clf_opt = SVC(kernel='linear', C=C_demo)
    clf_opt.fit(X_opt, y_opt)

    w = clf_opt.coef_[0]
    b = clf_opt.intercept_[0]

    # Data points
    mask_pos = y_opt == 1
    mask_neg = y_opt == -1

    fig_opt = go.Figure()

    fig_opt.add_trace(go.Scatter(
        x=X_opt[mask_pos, 0], y=X_opt[mask_pos, 1],
        mode='markers',
        marker=dict(size=10, color='#667eea'),
        name='Class +1'
    ))

    fig_opt.add_trace(go.Scatter(
        x=X_opt[mask_neg, 0], y=X_opt[mask_neg, 1],
        mode='markers',
        marker=dict(size=10, color='#f56565'),
        name='Class -1'
    ))

    # Support vectors
    sv = clf_opt.support_vectors_
    fig_opt.add_trace(go.Scatter(
        x=sv[:, 0], y=sv[:, 1],
        mode='markers',
        marker=dict(size=15, color='#ffd700', symbol='star',
                   line=dict(width=2, color='#ff8c00')),
        name=f'Support Vectors ({len(sv)})'
    ))

    # Decision boundary
    x_line = np.linspace(X_opt[:, 0].min()-1, X_opt[:, 0].max()+1, 100)
    y_line = -(w[0] * x_line + b) / w[1]

    fig_opt.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode='lines',
        line=dict(color='#48bb78', width=3),
        name='Decision Boundary'
    ))

    # Margins
    y_up = -(w[0] * x_line + b - 1) / w[1]
    y_down = -(w[0] * x_line + b + 1) / w[1]

    fig_opt.add_trace(go.Scatter(
        x=x_line, y=y_up,
        mode='lines',
        line=dict(color='#48bb78', width=2, dash='dash'),
        name='Upper Margin'
    ))

    fig_opt.add_trace(go.Scatter(
        x=x_line, y=y_down,
        mode='lines',
        line=dict(color='#48bb78', width=2, dash='dash'),
        name='Lower Margin'
    ))

    # Calculate margin width
    margin_width = 2 / np.linalg.norm(w)

    fig_opt.update_layout(
        height=500,
        showlegend=True,
        title=f"C = {C_demo} | Margin Width = {margin_width:.3f} | Support Vectors: {len(sv)}",
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray')
    )

    with col2:
        st.plotly_chart(fig_opt, use_container_width=True)

        # Show metrics
        st.markdown(f"""
        <div class='example-box'>
        <b>Current Metrics (C = {C_demo}):</b><br>
        • Support Vectors: {len(sv)}<br>
        • Margin Width: {margin_width:.4f}<br>
        • ||w||: {np.linalg.norm(w):.4f}<br>
        • Training Accuracy: {clf_opt.score(X_opt, y_opt)*100:.1f}%
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Visualization of margin maximization
    st.subheader("📏 Understanding Margin Maximization")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        ### Why maximize margin?

        The margin is the distance from the decision boundary to the nearest data point.
        """)

        st.latex(r"\text{Margin} = \frac{2}{||w||}")

        st.markdown("""
        - To maximize margin, we minimize ||w||
        - Larger margin → Better generalization to new data
        - The "maximum margin" classifier is unique!
        """)

        # Interactive visualization of different margins
        margin_scale = st.slider("🎚️ Adjust ||w|| manually", 0.5, 5.0, 2.0, 0.1)

        margin_dist = 1 / margin_scale

        st.markdown(f"""
        <div class='example-box'>
        <b>Current Values:</b><br>
        • ||w|| = {margin_scale:.2f}<br>
        • Margin = 2/||w|| = 2/{margin_scale:.2f} = <b>{2*margin_dist:.3f}</b>
        </div>
        """, unsafe_allow_html=True)

        fig_margin = go.Figure()

        # Sample points
        points_x = np.array([2, 3, 2.5, -2, -3, -2.5])
        points_y = np.array([2, 2.5, 1.5, -2, -2.5, -1.5])
        colors = ['#667eea']*3 + ['#f56565']*3

        for i in range(6):
            fig_margin.add_trace(go.Scatter(
                x=[points_x[i]], y=[points_y[i]],
                mode='markers',
                marker=dict(size=14, color=colors[i], line=dict(width=2, color='white')),
                showlegend=False
            ))

        # Decision boundary (always through origin for simplicity)
        x_line = np.linspace(-5, 5, 100)
        y_line = -x_line  # Simple diagonal line

        fig_margin.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode='lines',
            line=dict(color='#48bb78', width=4),
            name='Decision Boundary'
        ))

        # Margins based on slider
        offset = margin_dist * np.sqrt(2)

        fig_margin.add_trace(go.Scatter(
            x=x_line, y=y_line + offset,
            mode='lines',
            line=dict(color='#48bb78', width=2, dash='dash'),
            name='Margin'
        ))

        fig_margin.add_trace(go.Scatter(
            x=x_line, y=y_line - offset,
            mode='lines',
            line=dict(color='#48bb78', width=2, dash='dash'),
            showlegend=False
        ))

        fig_margin.update_layout(
            title=f"Margin Width = {2*margin_dist:.2f} (||w|| = {margin_scale:.2f})",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            height=400,
            plot_bgcolor='#f8f9fa',
            xaxis=dict(range=[-5, 5], gridcolor='lightgray'),
            yaxis=dict(range=[-5, 5], gridcolor='lightgray')
        )

    with col_b:
        st.plotly_chart(fig_margin, use_container_width=True)

        st.metric("Current Margin", f"{2*margin_dist:.3f}")
        st.metric("||w||", f"{margin_scale:.3f}")

        if margin_dist < 0.5:
            st.warning("⚠️ Small margin! Risk of overfitting.")
        elif margin_dist > 1.5:
            st.error("✗ Margin violates data points! Not feasible.")
        else:
            st.success("✓ Valid margin! SVM finds the maximum feasible margin.")

# Tab 6: Predictions
with tabs[6]:
    st.header("How SVM Makes Predictions")

    st.markdown("""
    ### The Decision Function

    For a new point x, SVM computes a **score**:
    """)

    st.latex(r"f(x) = w^T x + b")

    st.markdown("""
    <div class='example-box'>
    <b>Classification Rule:</b><br>
    • If f(x) > 0 → Predict Class +1<br>
    • If f(x) < 0 → Predict Class -1<br>
    • If f(x) = 0 → Point is on the decision boundary<br><br>

    <b>Confidence:</b><br>
    • |f(x)| represents confidence<br>
    • Larger |f(x)| = more confident prediction<br>
    • Points far from boundary have high |f(x)|
    </div>
    """, unsafe_allow_html=True)

    # Train a simple SVM for demonstration
    np.random.seed(42)
    X_pred = np.vstack([
        np.random.randn(25, 2) * 0.8 + [2, 2],
        np.random.randn(25, 2) * 0.8 + [-2, -2]
    ])
    y_pred = np.array([1]*25 + [-1]*25)

    clf_pred = SVC(kernel='linear', C=1.0)
    clf_pred.fit(X_pred, y_pred)

    w = clf_pred.coef_[0]
    b = clf_pred.intercept_[0]

    # Display SVM parameters prominently
    st.markdown("---")
    st.subheader("📋 Trained SVM Parameters")

    col_params1, col_params2, col_params3 = st.columns(3)

    with col_params1:
        st.markdown(f"""
        <div class='formula-box'>
        <b>Weight Vector w:</b><br>
        w = [{w[0]:.4f}, {w[1]:.4f}]<br><br>
        ||w|| = {np.linalg.norm(w):.4f}
        </div>
        """, unsafe_allow_html=True)

    with col_params2:
        st.markdown(f"""
        <div class='formula-box'>
        <b>Bias b:</b><br>
        b = {b:.4f}<br><br>
        Intercept term
        </div>
        """, unsafe_allow_html=True)

    with col_params3:
        st.markdown(f"""
        <div class='formula-box'>
        <b>Decision Function:</b><br>
        f(x) = {w[0]:.4f}×x₁<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ {w[1]:.4f}×x₂<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ ({b:.4f})
        </div>
        """, unsafe_allow_html=True)

    with st.expander("❓ Help: What are w and b?"):
        st.markdown(f"""
        **Weight vector w = [{w[0]:.4f}, {w[1]:.4f}]:**
        - Defines the **direction** perpendicular to the hyperplane
        - Larger ||w|| = {np.linalg.norm(w):.4f} means smaller margin
        - Each component tells how much that feature contributes

        **Bias b = {b:.4f}:**
        - Shifts the hyperplane away from origin
        - Positive b → hyperplane shifts in -w direction
        - Negative b → hyperplane shifts in +w direction

        **Together:** They define the hyperplane wᵀx + b = 0
        """)

    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🧮 Calculate Score Step-by-Step")

        # Manual point input
        st.markdown("**Enter a test point:**")
        test_x1 = st.number_input("x₁ coordinate", -5.0, 5.0, 2.5, 0.1, key="test_x1_pred")
        test_x2 = st.number_input("x₂ coordinate", -5.0, 5.0, 1.5, 0.1, key="test_x2_pred")

        # Calculate score step by step
        term1 = w[0] * test_x1
        term2 = w[1] * test_x2
        score_calc = term1 + term2 + b

        st.markdown(f"""
        <div class='example-box'>
        <b>Step-by-Step Calculation:</b><br><br>

        <b>Step 1:</b> w₁ × x₁<br>
        = {w[0]:.4f} × {test_x1:.2f}<br>
        = <b>{term1:.4f}</b><br><br>

        <b>Step 2:</b> w₂ × x₂<br>
        = {w[1]:.4f} × {test_x2:.2f}<br>
        = <b>{term2:.4f}</b><br><br>

        <b>Step 3:</b> Add bias<br>
        = {term1:.4f} + {term2:.4f} + ({b:.4f})<br>
        = <b>{score_calc:.4f}</b><br><br>

        <b>Final Score: f(x) = {score_calc:.4f}</b>
        </div>
        """, unsafe_allow_html=True)

        # Prediction
        if score_calc > 0:
            st.success(f"✓ f(x) = {score_calc:.4f} > 0 → **Predict Class +1**")
            st.metric("Confidence", f"{abs(score_calc):.4f}", help="Distance from decision boundary × ||w||")
        elif score_calc < 0:
            st.error(f"✓ f(x) = {score_calc:.4f} < 0 → **Predict Class -1**")
            st.metric("Confidence", f"{abs(score_calc):.4f}", help="Distance from decision boundary × ||w||")
        else:
            st.warning(f"⚠️ f(x) = 0 → **On decision boundary!**")

        # Calculate actual distance
        distance_to_boundary = abs(score_calc) / np.linalg.norm(w)

        st.markdown(f"""
        <div class='formula-box'>
        <b>Geometric Distance:</b><br>
        distance = |f(x)| / ||w||<br>
        = |{score_calc:.4f}| / {np.linalg.norm(w):.4f}<br>
        = <b>{distance_to_boundary:.4f}</b> units<br><br>
        This is the actual geometric distance from the point to the hyperplane!
        </div>
        """, unsafe_allow_html=True)

        with st.expander("❓ Help: Why divide by ||w|| for distance?"):
            st.markdown("""
            **The score f(x) is not the true distance!**

            - f(x) is a **scaled distance** (scaled by ||w||)
            - To get the **geometric distance**, divide by ||w||:

            **Formula:** distance = |f(x)| / ||w||

            **Why?** The hyperplane normal w can have any length. Dividing by ||w|| normalizes it to get true distance.

            **Example with your point:**
            - If we doubled w (and b), f(x) would double
            - But geometric distance stays the same!
            - Division by ||w|| ensures this consistency
            """)

        st.markdown("---")

        if st.button("➕ Add This Point to Plot", use_container_width=True):
            if 'test_points' not in st.session_state:
                st.session_state.test_points = []
            st.session_state.test_points.append([test_x1, test_x2])

        if st.button("🗑️ Clear All Test Points", use_container_width=True):
            st.session_state.test_points = []

    with col2:
        # Create decision function heatmap
        x_min, x_max = -5, 5
        y_min, y_max = -5, 5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        Z = clf_pred.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig_pred = go.Figure()

        # Heatmap of decision function
        fig_pred.add_trace(go.Heatmap(
            x=np.linspace(x_min, x_max, 100),
            y=np.linspace(y_min, y_max, 100),
            z=Z,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="f(x) score"),
            hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>Score: %{z:.2f}<extra></extra>'
        ))

        # Training data
        mask_pos = y_pred == 1
        mask_neg = y_pred == -1

        fig_pred.add_trace(go.Scatter(
            x=X_pred[mask_pos, 0], y=X_pred[mask_pos, 1],
            mode='markers',
            marker=dict(size=8, color='#667eea', line=dict(width=2, color='white')),
            name='Training: Class +1'
        ))

        fig_pred.add_trace(go.Scatter(
            x=X_pred[mask_neg, 0], y=X_pred[mask_neg, 1],
            mode='markers',
            marker=dict(size=8, color='#f56565', line=dict(width=2, color='white')),
            name='Training: Class -1'
        ))

        # Decision boundary
        x_line = np.linspace(x_min, x_max, 100)
        y_line = -(w[0] * x_line + b) / w[1]

        fig_pred.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode='lines',
            line=dict(color='#ffd700', width=4),
            name='Decision Boundary (f(x)=0)'
        ))

        # Add test points if any
        if 'test_points' in st.session_state and len(st.session_state.test_points) > 0:
            test_points_array = np.array(st.session_state.test_points)
            fig_pred.add_trace(go.Scatter(
                x=test_points_array[:, 0], y=test_points_array[:, 1],
                mode='markers',
                marker=dict(size=15, color='#ffd700', symbol='star',
                           line=dict(width=3, color='black')),
                name='Test Points ⭐'
            ))

        fig_pred.update_layout(
            title="Decision Function Heatmap",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            height=500,
            showlegend=True
        )

        st.plotly_chart(fig_pred, use_container_width=True)

    # Show predictions table
    if 'test_points' in st.session_state and len(st.session_state.test_points) > 0:
        st.subheader("📊 Test Point Predictions")

        predictions_data = []
        for i, point in enumerate(st.session_state.test_points, 1):
            score = clf_pred.decision_function([point])[0]
            pred_class = clf_pred.predict([point])[0]

            # Calculate step by step
            calc_str = f"{w[0]:.2f}×{point[0]:.2f} + {w[1]:.2f}×{point[1]:.2f} + ({b:.2f})"

            predictions_data.append({
                '#': i,
                'Point': f"[{point[0]:.2f}, {point[1]:.2f}]",
                'Calculation': calc_str,
                'Score f(x)': f"{score:.3f}",
                'Prediction': f"Class {pred_class:+d}",
                'Confidence': f"{abs(score):.3f}"
            })

        df_pred = pd.DataFrame(predictions_data)
        st.dataframe(df_pred, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Vector geometry explanation
    st.subheader("📐 Understanding the Geometry")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"""
        <div class='example-box'>
        <b>Current SVM Parameters:</b><br><br>

        • Weight vector: w = [{w[0]:.2f}, {w[1]:.2f}]<br>
        • Bias: b = {b:.2f}<br>
        • ||w|| = {np.linalg.norm(w):.2f}<br>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        ### How it works:

        1. **wᵀx**: Projects point x onto direction of w
        2. **+ b**: Shifts the decision boundary
        3. **Sign of result**: Determines which side of boundary

        **Key Insight:** The vector w is **perpendicular** to the decision boundary!
        """)

    with col_b:
        # Geometric visualization
        fig_geom = go.Figure()

        # Show w vector
        fig_geom.add_trace(go.Scatter(
            x=[0, w[0]*2], y=[0, w[1]*2],
            mode='lines+markers',
            line=dict(color='#9f7aea', width=5),
            marker=dict(size=12),
            name='w (normal to boundary)'
        ))

        # Decision boundary
        x_line = np.linspace(-3, 3, 100)
        y_line = -(w[0] * x_line + b) / w[1]

        fig_geom.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode='lines',
            line=dict(color='#48bb78', width=4),
            name='Decision Boundary'
        ))

        # Sample point and its projection
        sample_point = np.array([2.5, 1.5])
        score_sample = w @ sample_point + b

        fig_geom.add_trace(go.Scatter(
            x=[sample_point[0]], y=[sample_point[1]],
            mode='markers+text',
            marker=dict(size=15, color='#ffd700', symbol='star',
                       line=dict(width=2, color='black')),
            text=[f'f(x)={score_sample:.2f}'],
            textposition='top center',
            name=f'Test Point'
        ))

        # Projection line
        fig_geom.add_trace(go.Scatter(
            x=[0, sample_point[0]], y=[0, sample_point[1]],
            mode='lines',
            line=dict(color='orange', width=3, dash='dot'),
            name='Position vector'
        ))

        fig_geom.update_layout(
            title="Geometric Interpretation",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            height=400,
            plot_bgcolor='#f8f9fa',
            showlegend=True
        )

        st.plotly_chart(fig_geom, use_container_width=True)

# Tab 7: Nonlinear Decision Boundaries & Kernels - NEW!
with tabs[7]:
    st.header("Nonlinear Decision Boundaries & Kernel Functions")
    st.markdown("**Understanding when and why we need kernels**")

    st.markdown("""
    ### The Limitation of Linear SVM

    So far, we've been working with **linear decision boundaries** - straight lines (or hyperplanes) that separate classes.

    But what if your data looks like this?
    """)

    col_intro1, col_intro2 = st.columns([1, 1])

    with col_intro1:
        st.markdown("""
        **Linear SVM works well when:**
        - Data is linearly separable
        - A straight line can separate the classes
        - Simple decision boundaries suffice

        **Linear SVM fails when:**
        - Data has circular or curved patterns
        - Classes are nested within each other
        - Decision boundary needs to be nonlinear
        """)

    with col_intro2:
        # Show example of nonlinear data
        np.random.seed(42)
        X_circle, y_circle = make_circles(n_samples=100, noise=0.1, factor=0.5)

        fig_nonlin = go.Figure()

        # make_circles returns labels as 0 and 1
        mask_pos = y_circle == 1
        mask_neg = y_circle == 0

        fig_nonlin.add_trace(go.Scatter(
            x=X_circle[mask_pos, 0], y=X_circle[mask_pos, 1],
            mode='markers',
            marker=dict(size=10, color='#667eea'),
            name='Class +1'
        ))

        fig_nonlin.add_trace(go.Scatter(
            x=X_circle[mask_neg, 0], y=X_circle[mask_neg, 1],
            mode='markers',
            marker=dict(size=10, color='#f56565'),
            name='Class -1'
        ))

        fig_nonlin.update_layout(
            title="Circular Data - Linear SVM Will Fail!",
            xaxis_title="x₁",
            yaxis_title="x₂",
            height=400,
            plot_bgcolor='#f8f9fa'
        )

        st.plotly_chart(fig_nonlin, use_container_width=True)

    st.markdown("---")

    st.subheader("Solution: Transform to Higher Dimensions")

    st.markdown("""
    ### The Key Insight

    **Even if data is not linearly separable in 2D, it might be linearly separable in a higher dimension!**
    """)

    col_sol1, col_sol2 = st.columns([1, 1])

    with col_sol1:
        st.markdown("""
        **Example: The Circle Problem**

        In 2D: x = [x₁, x₂]
        - Red points (inner circle)
        - Blue points (outer ring)
        - **No straight line can separate them!**

        But if we add a new feature:
        """)

        st.latex(r"\phi(x) = [x_1, x_2, x_1^2 + x_2^2]")

        st.markdown("""
        Now we have 3D: x = [x₁, x₂, r²]
        - The third dimension is the distance from origin
        - Inner circle: small r²
        - Outer ring: large r²
        - **A horizontal plane can separate them!**
        """)

    with col_sol2:
        st.markdown("""
        <div class='example-box'>
        <b>The Transformation Process:</b><br><br>

        <b>Step 1:</b> Map 2D → 3D<br>
        φ: [x₁, x₂] → [x₁, x₂, x₁² + x₂²]<br><br>

        <b>Step 2:</b> Find linear separator in 3D<br>
        A plane: w₁x₁ + w₂x₂ + w₃(x₁² + x₂²) + b = 0<br><br>

        <b>Step 3:</b> Project back to 2D<br>
        This becomes a circle in 2D!<br><br>

        <b>Result:</b> Nonlinear boundary in original space!
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Enter Kernel Functions")

    col_kern1, col_kern2 = st.columns([1, 1])

    with col_kern1:
        st.markdown("""
        ### The Problem with Explicit Transformation

        Transforming to higher dimensions has issues:
        - **Computational cost**: Many new features to compute
        - **Memory**: Storing high-dimensional vectors
        - **Curse of dimensionality**: Performance degrades

        **Example:** Polynomial degree 5 with 10 features:
        - Original: 10 dimensions
        - Transformed: 2,002 dimensions!
        - That's 200× more features to store and compute!
        """)

    with col_kern2:
        st.markdown("""
        ### The Kernel Trick Solution

        **Key idea:** We don't actually need φ(x)!

        SVM only needs **dot products** between points:
        """)

        st.latex(r"\phi(x)^T \phi(z)")

        st.markdown("""
        A **kernel function** K(x, z) computes this directly:
        """)

        st.latex(r"K(x, z) = \phi(x)^T \phi(z)")

        st.markdown("""
        **No explicit transformation needed!**

        **Example - Polynomial Kernel:**
        """)

        st.latex(r"K(x, z) = (x^T z + 1)^d")

        st.markdown("""
        This computes the dot product in polynomial feature space
        **without ever computing φ(x)**!
        """)

    st.markdown("---")

    st.subheader("Common Kernels & When to Use Them")

    st.markdown("""
    <div class='example-box'>
    <b>1. Linear Kernel:</b> K(x, z) = xᵀz<br>
    • Use when: Data is already linearly separable<br>
    • Decision boundary: Straight line/hyperplane<br>
    • Fastest, simplest option<br><br>

    <b>2. Polynomial Kernel:</b> K(x, z) = (xᵀz + c)ᵈ<br>
    • Use when: Data has polynomial relationships<br>
    • Decision boundary: Polynomial curves<br>
    • Parameter d controls complexity<br>
    • Example: degree 2 for ellipses, degree 3 for cubic curves<br><br>

    <b>3. RBF (Gaussian) Kernel:</b> K(x, z) = exp(-γ||x - z||²)<br>
    • Use when: Unsure about data structure (most versatile!)<br>
    • Decision boundary: Smooth, flexible shapes<br>
    • Parameter γ controls smoothness<br>
    • Can fit very complex patterns<br>
    • Most commonly used in practice
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Interactive: See the Difference")

    kernel_choice = st.selectbox(
        "Choose a kernel to see how it handles circular data",
        ["Linear (will fail)", "RBF (will work)"],
        key="kernel_demo"
    )

    # Train SVM with selected kernel
    if "Linear" in kernel_choice:
        clf_demo = SVC(kernel='linear', C=1.0)
    else:
        clf_demo = SVC(kernel='rbf', gamma=1.0, C=1.0)

    clf_demo.fit(X_circle, y_circle)

    # Create decision boundary
    x_min, x_max = X_circle[:, 0].min() - 0.5, X_circle[:, 0].max() + 0.5
    y_min, y_max = X_circle[:, 1].min() - 0.5, X_circle[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    Z = clf_demo.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    accuracy = clf_demo.score(X_circle, y_circle) * 100

    # Create tabs for 2D and 3D views
    view_tabs = st.tabs(["📊 2D View (Original Space)", "🎲 3D View (Feature Space)"])

    with view_tabs[0]:
        # 2D visualization
        fig_demo = go.Figure()

        # Decision regions with better color for two classes
        fig_demo.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, 200),
            y=np.linspace(y_min, y_max, 200),
            z=Z,
            colorscale=[[0, 'rgba(245,101,101,0.4)'], [0.5, 'rgba(255,255,255,0.1)'],
                       [1, 'rgba(102,126,234,0.4)']],
            showscale=False,
            contours=dict(start=-3, end=3, size=0.5),
            hovertemplate='x₁: %{x:.2f}<br>x₂: %{y:.2f}<br>Score: %{z:.2f}<extra></extra>'
        ))

        # Data points - TWO DISTINCT CLASSES with better styling
        fig_demo.add_trace(go.Scatter(
            x=X_circle[mask_pos, 0], y=X_circle[mask_pos, 1],
            mode='markers',
            marker=dict(
                size=12,
                color='#667eea',
                line=dict(width=2, color='#4c51bf'),
                symbol='circle'
            ),
            name='Class +1 (Outer Ring)'
        ))

        fig_demo.add_trace(go.Scatter(
            x=X_circle[mask_neg, 0], y=X_circle[mask_neg, 1],
            mode='markers',
            marker=dict(
                size=12,
                color='#f56565',
                line=dict(width=2, color='#c53030'),
                symbol='circle'
            ),
            name='Class -1 (Inner Circle)'
        ))

        # Decision boundary (f(x) = 0)
        fig_demo.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, 200),
            y=np.linspace(y_min, y_max, 200),
            z=Z,
            showscale=False,
            contours=dict(start=0, end=0, coloring='lines'),
            line=dict(color='#48bb78', width=4),
            name='Decision Boundary'
        ))

        fig_demo.update_layout(
            title=f"{kernel_choice} - 2D View | Accuracy: {accuracy:.1f}%",
            xaxis_title="x₁",
            yaxis_title="x₂",
            height=500,
            plot_bgcolor='white',
            showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
        )

        st.plotly_chart(fig_demo, use_container_width=True)

    with view_tabs[1]:
        # 3D visualization - show how kernel implicitly transforms data
        if "RBF" in kernel_choice:
            st.markdown("""
            ### 🎲 3D Feature Space Visualization

            The RBF kernel **implicitly** maps data to an infinite-dimensional space!
            Here we show a 3D projection where the decision boundary becomes easier to see.

            The surface shows the **decision function** f(x) - notice how it creates separation between classes.
            """)

            # Create 3D surface plot
            fig_3d = go.Figure()

            # Decision function surface
            fig_3d.add_trace(go.Surface(
                x=np.linspace(x_min, x_max, 200),
                y=np.linspace(y_min, y_max, 200),
                z=Z,
                colorscale='RdBu',
                opacity=0.8,
                name='Decision Function f(x)',
                hovertemplate='x₁: %{x:.2f}<br>x₂: %{y:.2f}<br>f(x): %{z:.2f}<extra></extra>',
                showscale=True,
                colorbar=dict(title="f(x)", len=0.7)
            ))

            # Project data points to their f(x) values
            Z_points_pos = clf_demo.decision_function(X_circle[mask_pos])
            Z_points_neg = clf_demo.decision_function(X_circle[mask_neg])

            # Class +1 points (blue)
            fig_3d.add_trace(go.Scatter3d(
                x=X_circle[mask_pos, 0],
                y=X_circle[mask_pos, 1],
                z=Z_points_pos,
                mode='markers',
                marker=dict(
                    size=8,
                    color='#667eea',
                    line=dict(width=2, color='#4c51bf'),
                    symbol='circle'
                ),
                name='Class +1 (Outer Ring)'
            ))

            # Class -1 points (red)
            fig_3d.add_trace(go.Scatter3d(
                x=X_circle[mask_neg, 0],
                y=X_circle[mask_neg, 1],
                z=Z_points_neg,
                mode='markers',
                marker=dict(
                    size=8,
                    color='#f56565',
                    line=dict(width=2, color='#c53030'),
                    symbol='circle'
                ),
                name='Class -1 (Inner Circle)'
            ))

            # Add decision plane at z=0
            xx_plane, yy_plane = np.meshgrid(
                np.linspace(x_min, x_max, 20),
                np.linspace(y_min, y_max, 20)
            )
            zz_plane = np.zeros_like(xx_plane)

            fig_3d.add_trace(go.Surface(
                x=xx_plane,
                y=yy_plane,
                z=zz_plane,
                opacity=0.3,
                colorscale=[[0, '#48bb78'], [1, '#48bb78']],
                showscale=False,
                name='Decision Plane (f(x)=0)',
                hoverinfo='skip'
            ))

            fig_3d.update_layout(
                title=f"RBF Kernel - 3D Feature Space View | Accuracy: {accuracy:.1f}%",
                scene=dict(
                    xaxis_title="x₁",
                    yaxis_title="x₂",
                    zaxis_title="f(x) - Decision Function",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
                ),
                height=600,
                showlegend=True,
                legend=dict(x=0.7, y=0.9, bgcolor='rgba(255,255,255,0.8)')
            )

            st.plotly_chart(fig_3d, use_container_width=True)

            st.success("""
            ✓ **Notice:** Points are separated by the decision plane at f(x)=0!
            - Blue points (outer ring) are above the plane: f(x) > 0
            - Red points (inner circle) are below the plane: f(x) < 0
            - The RBF kernel made this separation possible!
            """)

        else:
            st.info("""
            ℹ️ **3D visualization is only available for RBF kernel**

            Linear kernel works in the original 2D space, so there's no higher-dimensional
            transformation to visualize.

            Select "RBF (will work)" to see the 3D feature space!
            """)

    # Metrics row
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Accuracy", f"{accuracy:.1f}%")
    with col_m2:
        st.metric("Support Vectors", len(clf_demo.support_vectors_))
    with col_m3:
        kernel_name = "Linear" if "Linear" in kernel_choice else "RBF"
        st.metric("Kernel Type", kernel_name)

    st.markdown("---")

    st.markdown("""
    ### Key Takeaways

    **Nonlinear data requires nonlinear decision boundaries:**
    - Linear SVM: Works only for linearly separable data
    - Kernel SVM: Can handle complex, nonlinear patterns

    **Kernel functions are the solution:**
    - They implicitly map data to higher dimensions
    - No need to compute explicit transformations
    - Much faster and more efficient

    **Common kernels:**
    - **Linear**: For linearly separable data
    - **Polynomial**: For polynomial patterns
    - **RBF**: For complex patterns (most versatile!)

    Proceed to the next tab to see the computational details of how kernels save time!
    """)

# Tab 8: Kernel Trick Deep Dive (renumbered)
with tabs[8]:
    st.header("⚡ Kernel Trick Deep Dive: Explicit vs Implicit")
    st.markdown("**See why the kernel trick is so powerful - with actual computations!**")

    st.markdown("""
    ### 🤔 The Problem

    For non-linear data, we need to transform features to higher dimensions:
    - **Explicit transformation**: Map x → φ(x), then compute φ(x)^T φ(z)
    - **Kernel trick**: Compute K(x, z) directly without φ(x)!

    Let's see the difference with **real calculations**! 👇
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📍 Input Two Points")

        st.markdown("**Point X:**")
        x1_k = st.number_input("x₁", -3.0, 3.0, 2.0, 0.5, key="x1_kernel")
        x2_k = st.number_input("x₂", -3.0, 3.0, 1.0, 0.5, key="x2_kernel")

        st.markdown("**Point Z:**")
        z1_k = st.number_input("z₁", -3.0, 3.0, 1.0, 0.5, key="z1_kernel")
        z2_k = st.number_input("z₂", -3.0, 3.0, 1.0, 0.5, key="z2_kernel")

        x_point = np.array([x1_k, x2_k])
        z_point = np.array([z1_k, z2_k])

        st.markdown(f"""
        <div class='example-box'>
        <b>Your Points:</b><br>
        x = [{x1_k}, {x2_k}]<br>
        z = [{z1_k}, {z2_k}]
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Polynomial Kernel (degree 2)")

        st.markdown("""
        We'll use a **polynomial kernel of degree 2**:
        """)

        st.latex(r"K(x, z) = (x^T z + 1)^2")

        st.markdown("""
        This is equivalent to mapping to a **6-dimensional space**!
        """)

    st.markdown("---")

    # Method 1: Explicit Transformation
    st.subheader("Method 1: 🐌 Explicit Feature Transformation (Slow Way)")

    col_m1_1, col_m1_2 = st.columns([1, 1])

    with col_m1_1:
        st.markdown("""
        **Step 1:** Transform each point to higher dimension

        For polynomial degree 2, the transformation is:
        """)

        st.latex(r"\phi(x) = [x_1^2, \sqrt{2}x_1 x_2, x_2^2, \sqrt{2}x_1, \sqrt{2}x_2, 1]")

        # Compute explicit transformations
        phi_x = np.array([
            x1_k**2,
            np.sqrt(2) * x1_k * x2_k,
            x2_k**2,
            np.sqrt(2) * x1_k,
            np.sqrt(2) * x2_k,
            1
        ])

        phi_z = np.array([
            z1_k**2,
            np.sqrt(2) * z1_k * z2_k,
            z2_k**2,
            np.sqrt(2) * z1_k,
            np.sqrt(2) * z2_k,
            1
        ])

        st.markdown(f"""
        <div class='formula-box'>
        <b>Transformed Features:</b><br><br>

        φ(x) = [{phi_x[0]:.2f}, {phi_x[1]:.2f}, {phi_x[2]:.2f}, {phi_x[3]:.2f}, {phi_x[4]:.2f}, {phi_x[5]:.2f}]<br><br>

        φ(z) = [{phi_z[0]:.2f}, {phi_z[1]:.2f}, {phi_z[2]:.2f}, {phi_z[3]:.2f}, {phi_z[4]:.2f}, {phi_z[5]:.2f}]
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        **Dimensions:**
        - Original space: **2D** (x₁, x₂)
        - Transformed space: **6D** (6 features!)
        """)

    with col_m1_2:
        st.markdown("""
        **Step 2:** Compute dot product in high-dimensional space
        """)

        st.latex(r"\phi(x)^T \phi(z) = \sum_{i=1}^{6} \phi(x)_i \times \phi(z)_i")

        # Compute dot product explicitly
        dot_product_explicit = phi_x @ phi_z

        st.markdown(f"""
        <div class='formula-box'>
        <b>Detailed Calculation:</b><br>
        = {phi_x[0]:.2f}×{phi_z[0]:.2f}<br>
        + {phi_x[1]:.2f}×{phi_z[1]:.2f}<br>
        + {phi_x[2]:.2f}×{phi_z[2]:.2f}<br>
        + {phi_x[3]:.2f}×{phi_z[3]:.2f}<br>
        + {phi_x[4]:.2f}×{phi_z[4]:.2f}<br>
        + {phi_x[5]:.2f}×{phi_z[5]:.2f}<br>
        ────────────────<br>
        = <b>{dot_product_explicit:.4f}</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='example-box'>
        <b>Computational Cost:</b><br>
        • Transform x: <b>6 operations</b><br>
        • Transform z: <b>6 operations</b><br>
        • Dot product: <b>6 multiplications + 5 additions</b><br>
        • <b>Total: ~17 operations</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Method 2: Kernel Trick
    st.subheader("Method 2: ⚡ Kernel Trick (Fast Way)")

    col_m2_1, col_m2_2 = st.columns([1, 1])

    with col_m2_1:
        st.markdown("""
        **One-step computation:**

        Use the kernel function directly:
        """)

        st.latex(r"K(x, z) = (x^T z + 1)^2")

        st.markdown("""
        **No need to compute φ(x) or φ(z)!**
        """)

        # Compute using kernel trick
        x_dot_z = x_point @ z_point
        kernel_result = (x_dot_z + 1) ** 2

        st.markdown(f"""
        <div class='formula-box'>
        <b>Step-by-Step:</b><br><br>

        1. x^T z = {x1_k}×{z1_k} + {x2_k}×{z2_k}<br>
        &nbsp;&nbsp;&nbsp;= {x1_k*z1_k:.2f} + {x2_k*z2_k:.2f}<br>
        &nbsp;&nbsp;&nbsp;= {x_dot_z:.2f}<br><br>

        2. (x^T z + 1)² = ({x_dot_z:.2f} + 1)²<br>
        &nbsp;&nbsp;&nbsp;= {x_dot_z + 1:.2f}²<br>
        &nbsp;&nbsp;&nbsp;= <b>{kernel_result:.4f}</b>
        </div>
        """, unsafe_allow_html=True)

    with col_m2_2:
        st.markdown(f"""
        <div class='example-box'>
        <b>Computational Cost:</b><br>
        • Dot product x^T z: <b>2 multiplications + 1 addition</b><br>
        • Add 1: <b>1 addition</b><br>
        • Square: <b>1 multiplication</b><br>
        • <b>Total: ~5 operations</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🎉 The Magic!")

        # Verify they're equal
        difference = abs(dot_product_explicit - kernel_result)

        if difference < 0.0001:
            st.success("✓ Both methods give the **SAME RESULT**!")
        else:
            st.error(f"⚠️ Difference: {difference:.6f}")

        st.markdown(f"""
        <div class='example-box'>
        <b>Results Comparison:</b><br>
        • Explicit: <b>{dot_product_explicit:.4f}</b><br>
        • Kernel Trick: <b>{kernel_result:.4f}</b><br>
        • Difference: <b>{difference:.6f}</b><br><br>

        <b>Speedup:</b> ~17 ops → ~5 ops<br>
        <b>~3.4× faster!</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Comparison for higher dimensions
    st.subheader("📊 Why This Matters: Higher Dimensions")

    col_comp1, col_comp2 = st.columns(2)

    with col_comp1:
        st.markdown("""
        ### Explicit Transformation Cost

        For polynomial kernel of degree d with n features:
        """)

        st.latex(r"\text{Dimension} = \binom{n+d}{d}")

        degrees = [2, 3, 4, 5]
        n_features = 10

        comparison_data = []
        for d in degrees:
            from math import comb
            dim = comb(n_features + d, d)
            explicit_ops = dim * 2 + dim  # transform both + dot product
            kernel_ops = n_features + 2  # dot product + power

            comparison_data.append({
                "Degree": d,
                "Dimensions": dim,
                "Explicit Ops": explicit_ops,
                "Kernel Ops": kernel_ops,
                "Speedup": f"{explicit_ops/kernel_ops:.1f}×"
            })

        df_comparison = pd.DataFrame(comparison_data)

        st.markdown("""
        **Example: 10 features, varying degree**
        """)

        st.dataframe(df_comparison, use_container_width=True, hide_index=True)

    with col_comp2:
        st.markdown("""
        ### 🚀 Key Takeaways

        **The Kernel Trick allows us to:**

        1. ✓ **Work in high-dimensional spaces** without computing φ(x)
        2. ✓ **Save massive computation** - exponential speedup!
        3. ✓ **Save memory** - don't store high-dimensional vectors
        4. ✓ **Enable infinite dimensions** (RBF kernel = ∞-D space!)

        **For degree 5 polynomial with 10 features:**
        - Explicit: Works in 2,002-D space (very slow!)
        - Kernel: Only ~12 operations (fast!)

        **This is why SVM with kernels is practical!** ⚡
        """)

        # Visual representation
        fig_speedup = go.Figure()

        fig_speedup.add_trace(go.Bar(
            x=[f"d={d}" for d in degrees],
            y=[row["Explicit Ops"] for row in comparison_data],
            name="Explicit Transform",
            marker_color='#f56565'
        ))

        fig_speedup.add_trace(go.Bar(
            x=[f"d={d}" for d in degrees],
            y=[row["Kernel Ops"] for row in comparison_data],
            name="Kernel Trick",
            marker_color='#48bb78'
        ))

        fig_speedup.update_layout(
            title="Operations Required: Explicit vs Kernel",
            xaxis_title="Polynomial Degree",
            yaxis_title="Number of Operations",
            barmode='group',
            height=300,
            yaxis_type='log'
        )

        st.plotly_chart(fig_speedup, use_container_width=True)

    st.markdown("---")

    # Interactive RBF example
    st.subheader("🌟 Bonus: RBF Kernel (Infinite Dimensions!)")

    st.markdown("""
    The **RBF (Radial Basis Function)** kernel maps to **infinite-dimensional** space:
    """)

    st.latex(r"K(x, z) = \exp(-\gamma ||x - z||^2)")

    gamma_rbf = st.slider("Gamma (γ)", 0.1, 2.0, 0.5, 0.1, key="gamma_rbf")

    distance_sq = np.sum((x_point - z_point)**2)
    rbf_result = np.exp(-gamma_rbf * distance_sq)

    col_rbf1, col_rbf2 = st.columns(2)

    with col_rbf1:
        st.markdown(f"""
        <div class='formula-box'>
        <b>RBF Calculation:</b><br><br>
        ||x - z||² = ({x1_k}-{z1_k})² + ({x2_k}-{z2_k})²<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {(x1_k-z1_k)**2:.2f} + {(x2_k-z2_k)**2:.2f}<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {distance_sq:.2f}<br><br>

        K(x,z) = exp(-{gamma_rbf}×{distance_sq:.2f})<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= exp({-gamma_rbf*distance_sq:.2f})<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <b>{rbf_result:.4f}</b>
        </div>
        """, unsafe_allow_html=True)

    with col_rbf2:
        st.markdown("""
        **Mind-blowing fact:**

        The RBF kernel corresponds to mapping into **infinite-dimensional space**!

        But we compute it with just:
        - 2 subtractions
        - 2 multiplications
        - 1 exponential

        **Try computing a dot product in infinite dimensions explicitly!** 🤯

        This is the true power of the kernel trick.
        """)

# Tab 9: Kernel Gallery
with tabs[9]:
    st.header("Kernel Gallery: Explore Different Kernels")

    st.markdown("""
    ### Visual Exploration of Kernel Functions

    Now that you understand **how** the kernel trick works (from the previous tab),
    let's explore **different kernels** and see them in action on real datasets!

    Try different kernels and datasets to see which works best! 🎨
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🎛️ Kernel Settings")

        kernel_type = st.selectbox(
            "Kernel Function",
            ["Linear", "Polynomial", "RBF (Radial Basis Function)"],
            help="Choose the kernel type for non-linear classification"
        )

        if kernel_type == "Polynomial":
            degree = st.slider("📊 Polynomial Degree", 2, 5, 3, 1)
            st.latex(f"K(x, z) = (x^T z + 1)^{{{degree}}}")
        elif kernel_type == "RBF (Radial Basis Function)":
            gamma = st.slider("🌊 Gamma (γ)", 0.1, 5.0, 1.0, 0.1,
                            help="Higher gamma = more curved decision boundary")
            st.latex(r"K(x, z) = e^{-\gamma ||x - z||^2}")
        else:
            st.latex(r"K(x, z) = x^T z")

        dataset_kernel = st.selectbox(
            "📈 Dataset Pattern",
            ["Circles", "Moons", "XOR-like"],
            help="Choose a non-linear dataset pattern"
        )

        st.markdown("""
        <div class='example-box'>
        <b>Common Kernels:</b><br><br>

        <b>1. Linear:</b> K(x,z) = xᵀz<br>
        → For linearly separable data<br><br>

        <b>2. Polynomial:</b> K(x,z) = (xᵀz + c)ᵈ<br>
        → Captures polynomial relationships<br><br>

        <b>3. RBF:</b> K(x,z) = exp(-γ||x-z||²)<br>
        → Most popular for non-linear data<br>
        → Creates circular/smooth boundaries
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Generate non-linear data
        np.random.seed(42)

        if dataset_kernel == "Circles":
            X_kernel, y_kernel = make_circles(n_samples=200, noise=0.1, factor=0.5)
        elif dataset_kernel == "Moons":
            X_kernel, y_kernel = make_moons(n_samples=200, noise=0.1)
        else:  # XOR-like
            X_kernel = np.random.randn(200, 2)
            y_kernel = np.logical_xor(X_kernel[:, 0] > 0, X_kernel[:, 1] > 0).astype(int) * 2 - 1

        # Map kernel selection
        if kernel_type == "Linear":
            kernel_name = 'linear'
            kernel_params = {}
        elif kernel_type == "Polynomial":
            kernel_name = 'poly'
            kernel_params = {'degree': degree}
        else:  # RBF
            kernel_name = 'rbf'
            kernel_params = {'gamma': gamma}

        # Train SVM with selected kernel
        clf_kernel = SVC(kernel=kernel_name, **kernel_params, C=1.0)
        clf_kernel.fit(X_kernel, y_kernel)

        # Create decision boundary
        x_min, x_max = X_kernel[:, 0].min() - 0.5, X_kernel[:, 0].max() + 0.5
        y_min, y_max = X_kernel[:, 1].min() - 0.5, X_kernel[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))

        Z = clf_kernel.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig_kernel = go.Figure()

        # Decision function heatmap
        fig_kernel.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, 200),
            y=np.linspace(y_min, y_max, 200),
            z=Z,
            colorscale='RdBu',
            showscale=True,
            colorbar=dict(title="Score"),
            contours=dict(
                start=-3,
                end=3,
                size=0.5,
            ),
            name='Decision Function'
        ))

        # Data points
        mask_pos = y_kernel == 1
        mask_neg = y_kernel == -1

        fig_kernel.add_trace(go.Scatter(
            x=X_kernel[mask_pos, 0], y=X_kernel[mask_pos, 1],
            mode='markers',
            marker=dict(size=6, color='#667eea', line=dict(width=1, color='white')),
            name='Class +1'
        ))

        fig_kernel.add_trace(go.Scatter(
            x=X_kernel[mask_neg, 0], y=X_kernel[mask_neg, 1],
            mode='markers',
            marker=dict(size=6, color='#f56565', line=dict(width=1, color='white')),
            name='Class -1'
        ))

        # Support vectors
        sv = clf_kernel.support_vectors_
        fig_kernel.add_trace(go.Scatter(
            x=sv[:, 0], y=sv[:, 1],
            mode='markers',
            marker=dict(size=10, color='#ffd700', symbol='star',
                       line=dict(width=2, color='#ff8c00')),
            name='Support Vectors ⭐'
        ))

        # Decision boundary (f(x) = 0)
        fig_kernel.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, 200),
            y=np.linspace(y_min, y_max, 200),
            z=Z,
            showscale=False,
            contours=dict(
                start=0,
                end=0,
                coloring='lines'
            ),
            line=dict(color='yellow', width=4),
            name='Decision Boundary'
        ))

        accuracy = clf_kernel.score(X_kernel, y_kernel) * 100

        fig_kernel.update_layout(
            title=f"{kernel_type} Kernel - Accuracy: {accuracy:.1f}%",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            height=500,
            showlegend=True,
            plot_bgcolor='white'
        )

        st.plotly_chart(fig_kernel, use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Accuracy", f"{accuracy:.1f}%")
        with col_b:
            st.metric("⭐ Support Vectors", len(sv))
        with col_c:
            st.metric("📊 Total Samples", len(X_kernel))

    st.markdown("---")

    # Comparison of kernels
    st.subheader("🔍 Kernel Comparison on Non-Linear Data")

    # Generate circular data for comparison
    X_comp, y_comp = make_circles(n_samples=100, noise=0.1, factor=0.5)

    kernels = [
        ('Linear', 'linear', {}),
        ('Polynomial (d=2)', 'poly', {'degree': 2}),
        ('Polynomial (d=3)', 'poly', {'degree': 3}),
        ('RBF (γ=1)', 'rbf', {'gamma': 1.0})
    ]

    fig_comp = make_subplots(
        rows=1, cols=4,
        subplot_titles=[k[0] for k in kernels],
        horizontal_spacing=0.08
    )

    for idx, (title, kernel, params) in enumerate(kernels, 1):
        clf = SVC(kernel=kernel, **params, C=1.0)
        clf.fit(X_comp, y_comp)

        # Create mesh
        x_min, x_max = X_comp[:, 0].min() - 0.5, X_comp[:, 0].max() + 0.5
        y_min, y_max = X_comp[:, 1].min() - 0.5, X_comp[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision regions
        fig_comp.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, 100),
            y=np.linspace(y_min, y_max, 100),
            z=Z,
            showscale=False,
            colorscale=[[0, 'rgba(245,101,101,0.3)'], [1, 'rgba(102,126,234,0.3)']],
            hoverinfo='skip'
        ), row=1, col=idx)

        # Plot points
        mask_pos = y_comp == 1
        mask_neg = y_comp == -1

        fig_comp.add_trace(go.Scatter(
            x=X_comp[mask_pos, 0], y=X_comp[mask_pos, 1],
            mode='markers',
            marker=dict(size=5, color='#667eea'),
            showlegend=False
        ), row=1, col=idx)

        fig_comp.add_trace(go.Scatter(
            x=X_comp[mask_neg, 0], y=X_comp[mask_neg, 1],
            mode='markers',
            marker=dict(size=5, color='#f56565'),
            showlegend=False
        ), row=1, col=idx)

        acc = clf.score(X_comp, y_comp) * 100
        fig_comp.add_annotation(
            text=f"<b>Acc: {acc:.0f}%</b>",
            xref=f"x{idx}", yref=f"y{idx}",
            x=0, y=y_min + 0.1,
            showarrow=False,
            font=dict(size=12, color="white"),
            bgcolor="#667eea",
            borderpad=5
        )

    fig_comp.update_layout(
        height=350,
        showlegend=False,
        title_text="How Different Kernels Handle Non-Linear Data"
    )

    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("""
    <div class='example-box'>
    <b>🎓 Key Takeaway:</b> Linear kernel fails on non-linear data, but polynomial and RBF kernels
    can capture complex patterns by implicitly mapping to higher-dimensional spaces!
    </div>
    """, unsafe_allow_html=True)

# Tab 10: Probability Estimation
with tabs[10]:
    st.header("Probability Estimation with Platt Scaling")

    st.markdown("""
    ### From Decision Scores to Probabilities

    SVMs naturally output **decision scores** f(x), not probabilities. But sometimes we need probability estimates!
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='formula-box'>
        <b>SVM Decision Function:</b><br>
        f(x) = wᵀx + b<br><br>

        <b>Output:</b><br>
        • f(x) > 0 → Class +1<br>
        • f(x) < 0 → Class -1<br><br>

        <b>Problem:</b><br>
        f(x) = 10 and f(x) = 2 both predict +1,<br>
        but we don't know how confident!
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='formula-box'>
        <b>Platt Scaling Solution:</b><br>
        P(y=1|x) = 1 / (1 + exp(A×f(x) + B))<br><br>

        <b>Output:</b><br>
        • Probability between 0 and 1<br>
        • P close to 1 → very confident +1<br>
        • P close to 0 → very confident -1<br>
        • P ≈ 0.5 → uncertain
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Interactive Platt Scaling Demo
    st.subheader("Interactive Probability Calculator")

    st.markdown("""
    The sigmoid function converts unbounded scores to probabilities in [0, 1].

    **Parameters A and B** are learned from training data using maximum likelihood estimation.
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Adjust Platt Parameters")
        A_platt = st.slider("A (slope parameter)", -5.0, 5.0, -1.0, 0.1,
                           help="Controls how steep the sigmoid curve is")
        B_platt = st.slider("B (shift parameter)", -5.0, 5.0, 0.0, 0.1,
                           help="Controls where the curve crosses 0.5 probability")

        st.markdown("### Test Decision Score")
        test_score = st.slider("f(x) - Decision Score", -5.0, 5.0, 1.5, 0.1,
                              help="The raw SVM decision function output")

    with col2:
        # Calculate probability
        z = A_platt * test_score + B_platt
        prob = 1 / (1 + np.exp(-z))

        st.markdown("### Calculation Steps")
        st.markdown(f"""
        <div class='example-box'>
        <b>Step 1:</b> Calculate sigmoid input<br>
        z = A × f(x) + B<br>
        z = {A_platt:.2f} × {test_score:.2f} + {B_platt:.2f}<br>
        z = {z:.4f}<br><br>

        <b>Step 2:</b> Apply sigmoid function<br>
        P(y=1|x) = 1 / (1 + exp(-z))<br>
        P(y=1|x) = 1 / (1 + exp(-{z:.4f}))<br>
        P(y=1|x) = {prob:.4f}<br><br>

        <b>Result:</b><br>
        <span style='font-size: 1.5rem; color: {"#48bb78" if prob > 0.5 else "#f56565"};'>
        P(y=1|x) = {prob*100:.2f}%
        </span>
        </div>
        """, unsafe_allow_html=True)

        # Prediction
        if prob > 0.5:
            st.success(f"✓ **Predict Class +1** with {prob*100:.1f}% confidence")
        else:
            st.error(f"✗ **Predict Class -1** with {(1-prob)*100:.1f}% confidence")

    # Visualization
    st.markdown("---")
    st.subheader("Sigmoid Curve Visualization")

    # Create range of scores
    scores_range = np.linspace(-5, 5, 100)
    probs_range = 1 / (1 + np.exp(-(A_platt * scores_range + B_platt)))

    fig_sigmoid = go.Figure()

    # Sigmoid curve
    fig_sigmoid.add_trace(go.Scatter(
        x=scores_range,
        y=probs_range,
        mode='lines',
        line=dict(color='#667eea', width=3),
        name='Probability Curve'
    ))

    # Current test point
    fig_sigmoid.add_trace(go.Scatter(
        x=[test_score],
        y=[prob],
        mode='markers+text',
        marker=dict(size=15, color='#f56565', symbol='star',
                   line=dict(width=2, color='white')),
        text=[f'f(x)={test_score:.2f}<br>P={prob:.3f}'],
        textposition='top center',
        name='Your Test Point'
    ))

    # Reference lines
    fig_sigmoid.add_hline(y=0.5, line_dash="dash", line_color="gray",
                         annotation_text="Decision Threshold (0.5)")
    fig_sigmoid.add_vline(x=0, line_dash="dash", line_color="gray",
                         annotation_text="Decision Boundary")

    # Uncertainty region
    fig_sigmoid.add_hrect(y0=0.4, y1=0.6, fillcolor="yellow", opacity=0.2,
                         annotation_text="Uncertain Region", annotation_position="left")

    fig_sigmoid.update_layout(
        title=f"Platt Scaling: Score → Probability (A={A_platt:.2f}, B={B_platt:.2f})",
        xaxis_title="Decision Score f(x)",
        yaxis_title="Probability P(y=1|x)",
        height=500,
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgray', range=[-5, 5]),
        yaxis=dict(gridcolor='lightgray', range=[0, 1])
    )

    st.plotly_chart(fig_sigmoid, use_container_width=True)

    # Help widget
    with st.expander("❓ Help: How are A and B determined?"):
        st.markdown("""
        **Training Process for Platt Scaling:**

        1. **Train SVM** on training data to get w and b

        2. **Get decision scores** f(xᵢ) for all training points

        3. **Fit sigmoid parameters** A and B using maximum likelihood:
           - Create target probabilities: tᵢ = 1 if yᵢ=+1, else 0
           - Find A and B that maximize:
             ```
             ∏ P(tᵢ|xᵢ) = ∏ [1/(1 + exp(A×f(xᵢ) + B))]^tᵢ
             ```

        4. **Use fitted A and B** to convert any future f(x) to probability

        **Intuition:**
        - **A (slope)**: How quickly probability changes with score
          - Large |A| → steep curve → confident predictions
          - Small |A| → gentle curve → more uncertainty

        - **B (shift)**: Where the 50% probability point is
          - B=0 → 50% at f(x)=0 (well-calibrated)
          - B≠0 → threshold shifted (calibration adjustment)

        **In scikit-learn:**
        ```python
        clf = SVC(kernel='rbf', probability=True)  # Enable Platt scaling
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)  # Get probabilities
        ```
        """)

    st.markdown("---")

    # Comparison example
    st.subheader("Score vs Probability: Comparison")

    st.markdown("""
    Let's see how different decision scores map to probabilities:
    """)

    # Sample scores
    sample_scores = [-3, -1.5, -0.5, 0, 0.5, 1.5, 3]
    sample_probs = [1 / (1 + np.exp(-(A_platt * s + B_platt))) for s in sample_scores]

    comparison_data = []
    for score, prob in zip(sample_scores, sample_probs):
        prediction = "Class +1" if prob > 0.5 else "Class -1"
        confidence = prob if prob > 0.5 else 1 - prob
        confidence_level = "Very High" if confidence > 0.9 else "High" if confidence > 0.75 else "Moderate" if confidence > 0.6 else "Low"

        comparison_data.append({
            "Score f(x)": f"{score:.2f}",
            "Raw Prediction": "+" if score > 0 else "-",
            "Probability P(y=1|x)": f"{prob:.4f} ({prob*100:.1f}%)",
            "Final Prediction": prediction,
            "Confidence": f"{confidence*100:.1f}% ({confidence_level})"
        })

    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)

    st.markdown("""
    <div class='example-box'>
    <b>Key Observations:</b><br>
    • Scores far from 0 (±3) → Very confident predictions<br>
    • Scores near 0 → Low confidence, uncertain predictions<br>
    • Probability provides nuanced view beyond binary +1/-1<br>
    • Useful for ranking predictions by confidence!
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # When to use probabilities
    st.subheader("When to Use Probability Estimates")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='formula-box'>
        <b>✓ Use Probabilities When:</b><br><br>

        • **Ranking predictions** by confidence<br>
        • **Setting custom thresholds** (not just 0.5)<br>
        • **Combining multiple models** (ensemble)<br>
        • **Cost-sensitive decisions** (medical, finance)<br>
        • **Explaining predictions** to stakeholders<br>
        • **Calibrating predictions** for better reliability
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='formula-box'>
        <b>✗ Raw Scores May Be Better When:</b><br><br>

        • **Only care about classification** (yes/no)<br>
        • **Speed is critical** (extra computation)<br>
        • **Small training data** (A, B unreliable)<br>
        • **Theoretical guarantees** matter (margin)<br>
        • **Direct optimization** of margin is goal
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Real example
    st.subheader("Real-World Example: Medical Diagnosis")

    st.markdown("""
    **Scenario:** Cancer screening using SVM

    Without probabilities:
    - Score = 0.1 → Predict "Cancer" (because > 0)
    - Score = 5.2 → Predict "Cancer" (because > 0)
    - **Problem:** Doctor doesn't know which patient needs urgent attention!

    With Platt scaling probabilities:
    - Score = 0.1 → P = 52% → "Possible cancer, monitor closely"
    - Score = 5.2 → P = 98% → "Very likely cancer, immediate action needed"
    - **Better:** Doctor can prioritize and make informed decisions!
    """)

    st.markdown("""
    <div class='example-box'>
    <b>🎓 Summary:</b><br><br>

    <b>Platt Scaling Formula:</b> P(y=1|x) = 1 / (1 + exp(A×f(x) + B))<br><br>

    <b>What it does:</b> Converts SVM scores to calibrated probabilities<br><br>

    <b>Parameters:</b> A and B learned from training data via maximum likelihood<br><br>

    <b>Benefit:</b> Provides confidence estimates and enables better decision-making<br><br>

    <b>Cost:</b> Requires additional training phase and validation data
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
border-radius: 15px; color: white; margin-top: 30px;'>
    <h3 style='color: white; margin: 10px 0;'>SEAS-8505 | Dr. Elbasheer | 1/17/2026</h3>
    <p style='font-size: 1.1rem; margin: 10px 0;'>🎓 Explore, Learn, and Master Support Vector Machines!</p>
    <p style='opacity: 0.9;'>Play with the controls, explore different datasets, and understand SVM visually</p>
</div>
""", unsafe_allow_html=True)
