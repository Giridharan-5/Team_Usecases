import streamlit as st
import pandas as pd
import requests
import plotly.express as px

API_BASE = "http://localhost:8000"


# -------------------- API HELPERS --------------------
def fetch_recommendations():
    try:
        r = requests.get(f"{API_BASE}/recommendations", timeout=50)
        r.raise_for_status()
        return r.json().get("recommendations", [])
    except Exception as e:
        st.error(f"Failed to fetch recommendations: {e}")
        return []


def trigger_pipeline(all_pipeline=False):
    endpoint = "/pipeline/all" if all_pipeline else "/pipeline"
    try:
        r = requests.post(f"{API_BASE}{endpoint}", timeout=5)
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, str(e)


# -------------------- UI HELPERS --------------------
def decision_badge(decision):
    if decision == "Scale Down":
        return "🟢 Scale Down"
    elif decision == "Scale Up":
        return "🔵 Scale Up"
    elif decision == "No Change":
        return "🟠 No Change"
    return decision


# -------------------- MAIN APP --------------------
def main():
    st.set_page_config(
        page_title="AI Rightsizing Dashboard",
        page_icon="📊",
        layout="wide"
    )

    st.title("🤖 AI Rightsizing Recommendations")
    st.markdown("---")

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.subheader("⚙️ Pipeline Controls")
      
        st.markdown("---")
        st.markdown(
            """
### 🎨 Decision Legend
🟢 **Scale Down**  
🔵 **Scale Up**  
🟠 **No Change**
"""
        )

    # ---------- DATA ----------
    recommendations = fetch_recommendations()

    if not recommendations:
        st.warning("No recommendations found. Run pipeline first.")
        st.info("POST http://localhost:8000/pipeline or /pipeline/all")
        return

    df = pd.DataFrame(recommendations)

    # ---------- METRICS ----------
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Recommendations", len(df))
    col2.metric("Scale Down", len(df[df["decision"] == "Scale Down"]))
    col3.metric("Scale Up", len(df[df["decision"] == "Scale Up"]))
    col4.metric("No Change", len(df[df["decision"] == "No Change"]))
    col5.metric("Avg Scale %", f"{df['scale_percent'].mean():.1f}%")

    st.markdown("---")

    # ---------- TABLE ----------
    st.subheader("📋 All Recommendations")

    df_display = df.copy()
    df_display["decision"] = df_display["decision"].apply(decision_badge)
    df_display["scale_percent"] = df_display["scale_percent"].round(1)

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "r_id": st.column_config.TextColumn("Resource ID"),
            "Server Name": st.column_config.TextColumn("Server Name"),
            "Resource Type": st.column_config.TextColumn("Resource Type"),
            "decision": st.column_config.TextColumn("Decision"),
            "avg_usage": st.column_config.ProgressColumn(
                "Average Usage (%)",
                min_value=0,
                max_value=100,
                format="%.1f%%"
            ),
            "max_usage": st.column_config.ProgressColumn(
                "Max Usage (%)",
                min_value=0,
                max_value=100,
                format="%.1f%%"
            ),
            "forecasted_max": st.column_config.ProgressColumn(
                "Forecasted Max (%)",
                min_value=0,
                max_value=100,
                format="%.1f%%"
            ),
            "scale_percent": st.column_config.NumberColumn(
                "Scale Percentage (%)",
                format="%.1f%%"
            ),
	    "itis_ticket_number": st.column_config.TextColumn("ITSM Ticket Number"),
            "action_status": st.column_config.TextColumn("Action Status"),
            "remarks": st.column_config.TextColumn("Remarks")
        },
    )

    # ---------- CHARTS ----------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Usage Distribution")
        fig1 = px.histogram(
            df,
            x="avg_usage",
            color="decision",
            nbins=20,
            title="Average Usage by Decision",
            labels={
                "avg_usage": "Average Usage (%)",
                "decision": "Decision"
            },
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("🎯 Scale Percentage by Resource")
        fig2 = px.scatter(
            df,
            x="r_id",
            y="scale_percent",
            color="decision",
            size="forecasted_max",
            hover_data=["Server Name", "Resource Type"],
            title="Scale Percentage Recommendations",
            labels={
                "r_id": "Resource ID",
                "scale_percent": "Scale Percentage (%)",
                "forecasted_max": "Forecasted Max Usage (%)",
                "decision": "Decision"
            },
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ---------- DOWNLOAD ----------
    st.markdown("---")
    st.download_button(
        "📥 Download CSV",
        df.to_csv(index=False),
        "recommendations.csv",
        "text/csv",
    )


if __name__ == "__main__":
    main()
