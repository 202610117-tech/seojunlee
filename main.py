import io, re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── CSS: 한글 폰트 ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ── CSV 읽기 (인코딩 이중 시도) ─────────────────────────────────────────────
@st.cache_data
def read_csv(file):
    if file is None:
        return pd.DataFrame()
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

# ── 열 이름 표준화 ───────────────────────────────────────────────────────────
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce"
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed
    return df

# ── 사분위수 계산 (IQR 방식) ─────────────────────────────────────────────────
def quartiles(series):
    v = series.dropna().sort_values().values
    n = len(v)
    if n < 4:
        return None, None, None
    def q(pos):
        idx = pos * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        f = idx - lo
        return v[lo] * (1 - f) + v[hi] * f
    return q(0.25), q(0.5), q(0.75)

# ── 막대 그래프 헬퍼 ─────────────────────────────────────────────────────────
def bar_metric_fig(labels, values, title, color="#4CAF50", avg_line=None):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker_color=color,
        text=[f"{v:.2f}" if isinstance(v, float) else str(v) for v in values],
        textposition="outside",
        name="값",
    ))
    if avg_line is not None:
        fig.add_hline(
            y=avg_line,
            line_dash="dot",
            line_color="red",
            annotation_text=f"평균 {avg_line:.2f}",
            annotation_position="top right",
        )
    fig.update_layout(
        title=title,
        xaxis_title="개체",
        yaxis_title="값",
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(t=50, r=20, l=50, b=50),
    )
    return fig

# ── 산점도 + 추세선 헬퍼 ─────────────────────────────────────────────────────
def scatter_with_trend(x, y, x_label, y_label):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 2:
        return None
    slope, intercept = np.polyfit(df["x"], df["y"], 1)
    r = np.corrcoef(df["x"], df["y"])[0, 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["x"], y=df["y"],
        mode="markers",
        marker=dict(size=8, color="#1976D2", opacity=0.7),
        name="데이터",
    ))
    x_line = np.array([df["x"].min(), df["x"].max()])
    fig.add_trace(go.Scatter(
        x=x_line,
        y=slope * x_line + intercept,
        mode="lines",
        line=dict(color="red", width=2, dash="dot"),
        name=f"추세선 (r={r:.3f})",
    ))
    fig.update_layout(
        title=f"{x_label} vs {y_label} (r = {r:.3f})",
        xaxis_title=x_label,
        yaxis_title=y_label,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(t=50, r=20, l=50, b=50),
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
#  사이드바
# ═══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("📁 CSV 업로드")
env_file   = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV", type="csv")

st.sidebar.markdown("---")
st.sidebar.subheader("📋 생육 측정 항목")
growth_cols = []
if growth_file is not None:
    gdf = std_cols(read_csv(growth_file))
    growth_cols = [c for c in gdf.columns if c not in ("개체번호", "time")]
    if "생중량" in growth_cols:
        growth_cols.remove("생중량")
    growth_cols.sort()
if growth_cols:
    selected = st.sidebar.multiselect(
        "분석할 측정 항목 (생중량 제외)",
        options=growth_cols,
        default=growth_cols[:2] if len(growth_cols) >= 2 else growth_cols,
    )
else:
    selected = []

# ── 데이터 로드 (한 번만) ────────────────────────────────────────────────────
with st.spinner("데이터를 불러오는 중..."):
    env_df   = std_cols(read_csv(env_file))   if env_file   is not None else pd.DataFrame()
    growth_df = std_cols(read_csv(growth_file)) if growth_file is not None else pd.DataFrame()

# 학교명 추출 (파일명 앞부분)
def school_name(filename):
    if filename is None:
        return "업로드"
    base = filename.rsplit(".", 1)[0]
    base = re.sub(r"[_-]+", " ", base)
    return base.strip() or "우리 학교"

school = school_name(env_file.name if env_file is not None else
                     growth_file.name if growth_file is not None else None)

st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")
st.title(f"🌱 {school} 극지식물 개체별 생육 차이 분석")

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — 실험 개요
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

with tab1:
    try:
        # 연구 배경 및 목적
        st.subheader("연구 배경 및 목적")
        st.markdown(f"""
**{school}**에서 재배 중인 극지식물의 **개체별 생육 차이를 분석**하는 대시보드입니다.

극지식물은 온도·습도·pH·EC 등 환경 조건에 민감하게 반응하므로,  
개체마다 생장 속도나 최종 생체량에 차이가 나타날 수 있습니다.

이 대시보드에서는 다음을 확인합니다:
- 개체별 **생중량 분포**와 차이 (평균·표준편차·변동계수 CV)
- **잎 수·지상부 길이**와 생중량의 관계 (상관계수·산점도)
- 생중량 **상위 25% vs 하위 25%** 개체 비교
        """.strip())

        st.markdown("---")
        st.subheader("데이터 요약")

        # 요약 표 구성
        rows = []
        n_growth = len(growth_df)
        if n_growth > 0:
            rows.append(["생육 결과 데이터", f"{n_growth}개체", "개별 개체 측정값"])
        if not env_df.empty:
            t = env_df["time"].dropna()
            rows.append(["환경 데이터", f"{len(env_df)}행", f"{t.min().date()} ~ {t.max().date()}"])
        if rows:
            st.table(pd.DataFrame(rows, columns=["데이터", "규모", "기간 / 내용"]))
        else:
            st.info("환경 데이터와 생육 결과 데이터를 업로드해 주세요.")

        st.markdown("---")
        st.subheader("주요 지표")

        # 지표 계산
        n_individuals = n_growth
        avg_temp  = env_df["temperature"].dropna().mean() if "temperature" in env_df.columns else None
        avg_hum   = env_df["humidity"].dropna().mean()      if "humidity"   in env_df.columns else None
        avg_fresh = growth_df["생중량"].dropna().mean()     if "생중량"     in growth_df.columns else None

        # 카드 4개 (2×2)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("총 개체수", f"{n_individuals}개체" if n_individuals > 0 else "데이터 없음")
            if avg_temp is not None:
                st.metric("평균 온도", f"{avg_temp:.1f}°C")
            else:
                st.info("환경 데이터가 없어 평균 온도를 표시할 수 없습니다.")
        with c2:
            st.metric("측정 기간",
                      f"{env_df['time'].min().date()} ~ {env_df['time'].max().date()}"
                      if not env_df.empty and "time" in env_df.columns
                      else "데이터 없음")
            if avg_fresh is not None:
                st.metric("평균 생중량", f"{avg_fresh:.2f}g")
            else:
                st.info("생중량 데이터가 없어 평균 생중량을 표시할 수 없습니다.")
        if avg_hum is not None:
            st.metric("평균 습도", f"{avg_hum:.1f}%")
        else:
            st.info("환경 데이터가 없어 평균 습도를 표시할 수 없습니다.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분만 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — 환경 데이터
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    try:
        if env_df.empty:
            st.info(f"🌡️ 환경 데이터 CSV를 업로드하면 이 탭의 내용이 표시됩니다.")
        else:
            st.subheader("🌡️ 환경 데이터 요약")

            env_cols = [c for c in ["temperature", "humidity", "ph", "ec"]
                        if c in env_df.columns]
            if env_cols:
                desc = env_df[env_cols].describe().T
                desc = desc[["mean", "min", "max", "std"]]
                desc.columns = ["평균", "최소", "최대", "표준편차"]
                st.dataframe(desc, width="stretch")
            else:
                st.info("환경 데이터에 인식되는 숫자 열(temperature, humidity, ph, ec)이 없습니다.")

            st.markdown("---")
            st.subheader("📈 날짜별 환경 변화 (일별 평균)")

            if "time" in env_df.columns:
                env_df_valid = env_df.dropna(subset=["time"]).copy()
                env_df_valid["date"] = env_df_valid["time"].dt.date

                env_num_cols = [c for c in ["temperature", "humidity", "ph", "ec"]
                                if c in env_df_valid.columns]

                if env_num_cols:
                    daily = env_df_valid.groupby("date")[env_num_cols].mean().reset_index()

                    n_plots = len(env_num_cols)
                    fig = make_subplots(
                        rows=n_plots, cols=1,
                        subplot_titles=[
                            "온도 변화 (°C)" if c == "temperature" else
                            "습도 변화 (%)"  if c == "humidity"   else
                            f"{c.upper()} 변화" if c in ("ph", "ec") else c
                            for c in env_num_cols
                        ],
                        vertical_spacing=0.12,
                    )
                    colors = {
                        "temperature": "#E53935",
                        "humidity":   "#43A047",
                        "ph":         "#7B1FA2",
                        "ec":         "#00838F",
                    }
                    for i, col in enumerate(env_num_cols):
                        fig.add_trace(
                            go.Scatter(
                                x=daily["date"], y=daily[col],
                                mode="lines",
                                name=col,
                                line=dict(color=colors.get(col, "#1976D2"), width=2),
                                fill="tozeroy",
                                fillcolor=f"rgba({255 if col=='temperature' else 0},"
                                          f"{150 if col=='humidity' else 0},"
                                          f"{50 if col=='ph' else 200},0.1)",
                            ),
                            row=i+1, col=1,
                        )
                    fig.update_layout(
                        height=max(200, n_plots * 180),
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        margin=dict(t=50, r=20, l=50, b=50),
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("환경 데이터에 온도·습도·pH·EC 열이 없어 시계열을 그릴 수 없습니다.")
            else:
                st.info("환경 데이터에 time 열이 없어 날짜별 시계열을 그릴 수 없습니다.")

            with st.expander("📋 환경 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(env_df, width="stretch")
                csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"{school}_환경데이터.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분만 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — 생육 결과
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    try:
        if growth_df.empty:
            st.info(f"📊 생육 결과 CSV를 업로드하면 이 탭의 내용이 표시됩니다.")
        else:
            # ── 핵심 결과 카드 ──────────────────────────────────────────────────
            st.subheader("🥇 핵심 결과")

            fresh = growth_df["생중량"].dropna()
            if len(fresh) > 0:
                avg_f = fresh.mean()
                max_idx = fresh.idxmax()
                min_idx = fresh.idxmin()

                # 개체는 원래 값 그대로 (정수/문자열)
                max_id = growth_df.loc[max_idx, "개체번호"]
                min_id = growth_df.loc[min_idx, "개체번호"]
                max_val = fresh.max()
                min_val = fresh.min()

                # 상위·하위 25%
                q1, q2, q3 = quartiles(fresh)
                top_mask   = fresh >= q3 if q3 is not None else fresh >= fresh.quantile(0.75)
                bot_mask   = fresh <= q1 if q1 is not None else fresh <= fresh.quantile(0.25)
                n_top = int(top_mask.sum())
                n_bot = int(bot_mask.sum())

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("평균 생중량", f"{avg_f:.2f}g")
                c2.metric("최대 개체", f"개체 {max_id} ({max_val:.2f}g)")
                c3.metric("최소 개체", f"개체 {min_id} ({min_val:.2f}g)")
                c4.metric("개체 수", f"{len(fresh)}개체")

                st.info(f"📌 생중량 상위 25%({n_top}개체): 평균 {fresh[top_mask].mean():.2f}g  |  "
                        f"하위 25%({n_bot}개체): 평균 {fresh[bot_mask].mean():.2f}g")
            else:
                st.info("생중량 데이터가 없습니다.")

            st.markdown("---")

            # ── 개체별 생육 비교 막대 ──────────────────────────────────────────
            st.subheader("📊 개체별 생육 비교")

            bar_cols = [c for c in ["생중량", "잎 수", "지상부 길이", "지하부 길이"]
                        if c in growth_df.columns]
            if bar_cols:
                n_bar = len(bar_cols)
                fig_bar = make_subplots(
                    rows=n_bar, cols=1,
                    subplot_titles=[
                        "생중량 (g)" if c == "생중량" else
                        "잎 수 (장)" if c == "잎 수" else
                        "지상부 길이 (mm)" if c == "지상부 길이" else
                        "지하부 길이 (mm)"
                        for c in bar_cols
                    ],
                    vertical_spacing=0.12,
                )
                bar_colors = {
                    "생중량":     "#E53935",
                    "잎 수":      "#43A047",
                    "지상부 길이": "#1E88E5",
                    "지하부 길이": "#7B1FA2",
                }
                for i, col in enumerate(bar_cols):
                    vals = growth_df[col].dropna()
                    if len(vals) == 0:
                        continue
                    avg_v = vals.mean()
                    is_top_cols = (col == "생중량")
                    if is_top_cols and q3 is not None:
                        colors_b = ["#E53935" if v >= q3 else
                                    "#C8E6C9" if v >= q2 else
                                    "#FFECB3" if v >= q1 else "#FFCDD2"
                                    for v in vals]
                    else:
                        colors_b = bar_colors.get(col, "#1976D2")
                    ids = growth_df.loc[vals.index, "개체번호"].values
                    fig_bar.add_trace(
                        go.Bar(
                            x=ids.astype(str),
                            y=vals.values,
                            marker_color=colors_b if isinstance(colors_b, list) else colors_b,
                            name=col,
                            hovertemplate=(f"{col}<br>개체: %{{x}}<br>값: %{{y:.2f}}<extra></extra>"
                                          if col in ("생중량", "잎 수") else
                                          f"{col}<br>개체: %{{x}}<br>값: %{{y:.1f}}<extra></extra>"),
                        ),
                        row=i+1, col=1,
                    )
                    if is_top_cols:
                        fig_bar.add_hline(
                            y=q3, line_dash="dot", line_color="orange",
                            annotation_text="상위 25% 기준",
                            annotation_position="top right",
                            row=i+1, col=1,
                        )
                fig_bar.update_layout(
                    height=max(250, n_bar * 170),
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    margin=dict(t=50, r=20, l=60, b=50),
                )
                st.plotly_chart(fig_bar, width="stretch")
            else:
                st.info("생육 데이터에 인식할 수 있는 측정 항목이 없습니다.")

            st.markdown("---")

            # ── 생중량 분포 (박스플롯) ──────────────────────────────────────────
            st.subheader("📦 생중량 분포")

            if "생중량" in growth_df.columns:
                fresh_vals = growth_df["생중량"].dropna()
                if len(fresh_vals) >= 2:
                    quant = fresh_vals.quantile([0.25, 0.5, 0.75])

                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(
                        y=fresh_vals.values,
                        name="생중량",
                        boxpoints="outliers",
                        marker=dict(color="#1976D2", size=6),
                        line=dict(color="#1565C0"),
                        fillcolor="rgba(25, 118, 212, 0.25)",
                        hovertemplate="생중량: %{y:.2f}g<extra></extra>",
                    ))
                    # 분위수 선 추가
                    for q_val, label, color in [
                        (quant[0.25], "Q1 (하위 25%)", "orange"),
                        (quant[0.5],  "Q2 (중앙값)",   "green"),
                        (quant[0.75], "Q3 (상위 25%)", "red"),
                    ]:
                        fig_box.add_hline(
                            y=q_val, line_dash="dot", line_color=color,
                            annotation_text=label,
                            annotation_position="top right",
                            line_width=1.5,
                        )
                    fig_box.update_layout(
                        title=f"생중량 분포 (n={len(fresh_vals)}개체)",
                        yaxis_title="생중량 (g)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        margin=dict(t=50, r=20, l=50, b=50),
                    )
                    st.plotly_chart(fig_box, width="stretch")

                    # CV + stats 카드
                    std_f = fresh_vals.std()
                    cv_f  = (std_f / avg_f * 100) if avg_f > 0 else None
                    stats_rows = [
                        ["평균",   f"{avg_f:.2f}g"],
                        ["표준편차", f"{std_f:.2f}g"],
                        ["변동계수(CV)",
                         f"{cv_f:.1f}%" if cv_f is not None else "계산 불가"],
                        ["중앙값", f"{quant[0.5]:.2f}g"],
                        ["Q1 (하위 25%)", f"{quant[0.25]:.2f}g"],
                        ["Q3 (상위 25%)", f"{quant[0.75]:.2f}g"],
                    ]
                    st.table(pd.DataFrame(stats_rows, columns=["통계", "값"]))
                else:
                    st.info("생중량 데이터가 2개 미만으로 분포를 표시할 수 없습니다.")
            else:
                st.info("생중량 열이 없어 분포 그래프를 그릴 수 없습니다.")

            st.markdown("---")

            # ── 상관관계 분석 (산점도 2개) ──────────────────────────────────────
            st.subheader("🔬 상관관계 분석")

            if "생중량" in growth_df.columns:
                fresh_s = growth_df["생중량"].dropna()
                pairs = [
                    ("잎 수",          "잎 수"),
                    ("지상부 길이",   "지상부 길이"),
                ]
                for x_col, x_label in pairs:
                    if x_col in growth_df.columns:
                        x_s = growth_df[x_col].dropna()
                        common_idx = fresh_s.index.intersection(x_s.index)
                        if len(common_idx) >= 2:
                            fig_sc = scatter_with_trend(
                                growth_df.loc[common_idx, x_col],
                                growth_df.loc[common_idx, "생중량"],
                                x_label, "생중량 (g)",
                            )
                            if fig_sc is not None:
                                st.plotly_chart(fig_sc, width="stretch")
                        else:
                            st.info(f"{x_label}과 생중량의 공통 데이터가 2개 미만이라 산점도를 그릴 수 없습니다.")
                    else:
                        st.info(f"{x_label} 열이 없어 산점도를 그릴 수 없습니다.")
            else:
                st.info("생중량 열이 없어 상관관계 분석을 할 수 없습니다.")

            with st.expander("📋 생육 데이터 원본 + CSV 다운로드"):
                st.dataframe(growth_df, width="stretch")
                csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"{school}_생육결과데이터.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분만 데이터 형식 때문에 표시하지 못했어요: {e}")
