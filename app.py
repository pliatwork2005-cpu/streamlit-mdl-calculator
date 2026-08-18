import hashlib
import reportlab
import pkgutil
import inspect

# --- DIRECT FIX FOR REPORTLAB ON PYTHON 3.8 ---
def _safe_md5(string=b'', **kwargs):
    kwargs.pop('usedforsecurity', None)
    return hashlib.md5(string)

for importer, modname, ispkg in pkgutil.walk_packages(reportlab.__path__, reportlab.__name__ + '.'):
    try:
        mod = __import__(modname, fromlist=['*'])
        if hasattr(mod, 'md5'):
            mod.md5 = _safe_md5
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            if hasattr(obj, '__globals__') and 'md5' in obj.__globals__:
                obj.__globals__['md5'] = _safe_md5
    except Exception:
        pass
# ---------------------------------------------

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Page configuration
st.set_page_config(page_title="Calibration & MDL Calculator", layout="wide")

st.title("Signal Slope, Standard Curve & MDL Analysis App")
st.markdown("Upload your **Standard Curve file** and **Blank Data file(s)** (CSV/TXT format), configure your report settings, then click **Start Analysis**.")

# Custom report file name and unit input boxes side by side
setting_col1, setting_col2 = st.columns(2)
with setting_col1:
    report_filename = st.text_input("Enter Report File Name", value="MDL_Analysis_Report")
with setting_col2:
    unit = st.text_input("Enter Concentration Unit (e.g., mg/L, ppb, nM)", value="mg/L")

# Layout for file uploaders
col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Standard Curve Data")
    std_file = st.file_uploader("Upload Standard Data", type=["csv", "txt"], key="std")

with col2:
    st.subheader("2. Blank Data (Concentration = 0)")
    blank_files = st.file_uploader(
        "Upload Blank Data (Multiple files allowed)",
        type=["csv", "txt"],
        accept_multiple_files=True,
        key="blank"
    )

st.markdown("---")

# Start Analysis Button
if st.button("🚀 Start Analysis", type="primary"):
    if std_file is None:
        st.error("Please upload a Standard Curve file before starting analysis.")
    else:
        # Process Standard Data
        content_std = std_file.getvalue().decode('utf-8', errors='ignore')
        sep_std = '\t' if '\t' in content_std else ','
        df_std = pd.read_csv(io.StringIO(content_std), header=None, sep=sep_std)

        num_cols = df_std.shape[1]
        results = []

        # Iterate through column pairs: (Col 0, Col 1), (Col 2, Col 3), etc.
        for i in range(0, num_cols - 1, 2):
            conc_val = df_std.iloc[0, i]
            if pd.isna(conc_val):
                conc_val = df_std.iloc[0, i + 1]

            try:
                concentration = float(conc_val)
            except (ValueError, TypeError):
                continue  # Skip if header is not a valid number

            # Extract time and signal data from row 1 onwards
            time_data = pd.to_numeric(df_std.iloc[1:, i], errors='coerce').values
            signal_data = pd.to_numeric(df_std.iloc[1:, i + 1], errors='coerce').values

            # Remove missing values
            valid_mask = ~np.isnan(time_data) & ~np.isnan(signal_data)
            t = time_data[valid_mask]
            s = signal_data[valid_mask]

            if len(t) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(t, s)
                results.append({
                    "Concentration": concentration,
                    "Slope": slope,
                    "Pair_Index": i // 2
                })

        if results:
            df_results = pd.DataFrame(results)

            # 1. Table of concentration to corresponding slope, sorted high to low
            df_sorted = df_results.sort_values(by="Concentration", ascending=False).reset_index(drop=True)

            df_sorted_display = df_sorted.copy()
            df_sorted_display["Slope"] = df_sorted_display["Slope"].apply(lambda x: f"{x:.3e}")
            df_sorted_display.rename(columns={"Concentration": f"Concentration ({unit})"}, inplace=True)

            st.header("Step 1: Individual Replicate Slopes")
            st.markdown("Showing individual slopes sorted from high to low concentrations:")
            st.dataframe(df_sorted_display[[f"Concentration ({unit})", "Slope"]])

            # 2. Average slopes of same concentration
            df_avg = df_sorted.groupby("Concentration")["Slope"].agg(['mean', 'std', 'count']).reset_index()
            df_avg = df_avg.sort_values(by="Concentration", ascending=False).reset_index(drop=True)
            df_avg.rename(columns={'mean': 'Average Slope', 'std': 'Std Dev', 'count': 'Replicates'}, inplace=True)

            df_avg_display = df_avg.copy()
            df_avg_display["Average Slope"] = df_avg_display["Average Slope"].apply(lambda x: f"{x:.3e}")
            df_avg_display["Std Dev"] = df_avg_display["Std Dev"].apply(lambda x: f"{x:.3e}")
            df_avg_display.rename(columns={"Concentration": f"Concentration ({unit})"}, inplace=True)

            st.header("Step 2: Average Slopes & Standard Curve")
            st.dataframe(df_avg_display[[f"Concentration ({unit})", "Average Slope", "Std Dev", "Replicates"]])

            # Linear Regression: Concentration (X) vs Average Slope (Y) using raw high-precision values
            x = df_avg["Concentration"].values
            y = df_avg["Average Slope"].values

            m, c, r_val, p_val, std_err = stats.linregress(x, y)
            r_squared = r_val ** 2

            st.success(f"**Standard Curve Equation:** Slope = ({m:.3e} * Concentration [{unit}]) + ({c:.3e})  |  (R² = {r_squared:.4f})")

            # Plot standard curve graph
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(x, y, color='blue', s=50, label='Average Slopes', zorder=3)
            x_line = np.linspace(min(x), max(x), 100)
            y_line = m * x_line + c
            ax.plot(x_line, y_line, color='red', linestyle='--', label=f'Fit: y = {m:.3e}x + {c:.3e}', zorder=2)
            ax.set_xlabel(f"Concentration ({unit})")
            ax.set_ylabel("Average Slope")
            ax.set_title("Standard Calibration Curve")
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend()
            st.pyplot(fig)

            # 3. Blank Data & MDL Calculation (Supporting Multiple Files)
            blank_slopes = []
            if blank_files:
                for blank_file in blank_files:
                    content_blank = blank_file.getvalue().decode('utf-8', errors='ignore')
                    sep_blank = '\t' if '\t' in content_blank else ','
                    df_blank = pd.read_csv(io.StringIO(content_blank), header=None, sep=sep_blank)

                    num_blank_cols = df_blank.shape[1]
                    for i in range(0, num_blank_cols - 1, 2):
                        time_b = pd.to_numeric(df_blank.iloc[1:, i], errors='coerce').values
                        signal_b = pd.to_numeric(df_blank.iloc[1:, i + 1], errors='coerce').values

                        valid_mask_b = ~np.isnan(time_b) & ~np.isnan(signal_b)
                        tb = time_b[valid_mask_b]
                        sb = signal_b[valid_mask_b]

                        if len(tb) > 1:
                            slope_b, _, _, _, _ = stats.linregress(tb, sb)
                            blank_slopes.append(slope_b)

            if blank_slopes:
                mean_blank = np.mean(blank_slopes)
                std_blank = np.std(blank_slopes, ddof=1) if len(blank_slopes) > 1 else 0.0

                limit = mean_blank + (3 * std_blank)
                mdl = (limit - c) / m if m != 0 else np.nan

                st.header("Step 3: Blank Analysis & Method Detection Limit (MDL)")
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Average Blank Slope", f"{mean_blank:.3e}")
                m_col2.metric("Blank Standard Deviation", f"{std_blank:.3e}")
                m_col3.metric("Critical Limit (Mean + 3×SD)", f"{limit:.3e}")
                m_col4.metric(f"Calculated MDL ({unit})", f"{mdl:.3e}")

                # --- PDF REPORT GENERATION ---
                pdf_buffer = io.BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
                story = []
                styles = getSampleStyleSheet()

                # Custom styles
                title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor('#1f77b4'), spaceAfter=10)
                h2_style = ParagraphStyle('H2Style', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#333333'), spaceBefore=10, spaceAfter=6)
                normal_style = ParagraphStyle('NormalStyle', parent=styles['Normal'], fontSize=10, leading=14)

                # Title
                story.append(Paragraph("Calibration & MDL Analysis Report", title_style))
                story.append(Paragraph(f"<b>Concentration Unit:</b> {unit}", normal_style))
                story.append(Spacer(1, 5))

                # Section 1: Standard Curve
                story.append(Paragraph("1. Standard Curve Regression", h2_style))
                story.append(Paragraph(f"<b>Equation:</b> Slope = ({m:.3e} * Concentration [{unit}]) + ({c:.3e})", normal_style))
                story.append(Paragraph(f"<b>R-squared (R²):</b> {r_squared:.4f}", normal_style))
                story.append(Spacer(1, 5))

                # Section 2: Blank & MDL
                story.append(Paragraph("2. Blank Analysis & Method Detection Limit (MDL)", h2_style))
                story.append(Paragraph(f"<b>Blank Replicates Analyzed:</b> {len(blank_slopes)}", normal_style))
                story.append(Paragraph(f"<b>Average Blank Slope:</b> {mean_blank:.3e} | <b>Standard Deviation:</b> {std_blank:.3e}", normal_style))
                story.append(Paragraph(f"<b>Critical Limit (Mean + 3×SD):</b> {limit:.3e}", normal_style))
                story.append(Paragraph(f"<b>Calculated MDL:</b> {mdl:.4e} {unit}", normal_style))
                story.append(Spacer(1, 8))

                # Section 3: Averaged Concentration Table
                story.append(Paragraph("3. Averaged Slopes per Concentration", h2_style))
                table_data = [[f"Concentration ({unit})", "Average Slope", "Std Dev", "Replicates"]]
                for _, row in df_avg.iterrows():
                    table_data.append([
                        str(row['Concentration']),
                        f"{row['Average Slope']:.3e}",
                        f"{row['Std Dev']:.3e}",
                        str(row['Replicates'])
                    ])

                t = Table(table_data, colWidths=[110, 160, 160, 110])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f0f2f6')),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.black),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,-1), 9),
                    ('BOTTOMPADDING', (0,0), (-1,0), 6),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey)
                ]))
                story.append(t)
                story.append(Spacer(1, 10))

                # Section 4: Plot Image
                story.append(Paragraph("4. Standard Calibration Curve Plot", h2_style))
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=200)
                img_buf.seek(0)
                story.append(RLImage(img_buf, width=420, height=260))

                doc.build(story)
                pdf_data = pdf_buffer.getvalue()
                pdf_buffer.close()

                # Format filename with .pdf extension
                if not report_filename.endswith('.pdf'):
                    final_filename = report_filename + ".pdf"
                else:
                    final_filename = report_filename

                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_data,
                    file_name=final_filename,
                    mime="application/pdf"
                )
            elif blank_files:
                st.warning("Could not extract slopes from the provided blank files. Please check formatting.")
        else:
            st.error("Could not parse concentrations or columns from the standard curve file.")
