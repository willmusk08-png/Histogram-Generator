

import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Histogram Fitter", page_icon="📊")

def parse_text_data(text):
    if not text:
        return np.array([])
    try:
        tokens = text.replace(",", " ").split()
        vals = [float(t) for t in tokens]
        return np.array(vals)
    except Exception:
        return np.array([])

def read_csv_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, header=None)
        arr = pd.to_numeric(df.stack(), errors='coerce').dropna().values
        return arr.astype(float)
    except Exception:
        try:
            s = pd.read_csv(uploaded_file).iloc[:, 0]
            return pd.to_numeric(s, errors='coerce').dropna().values.astype(float)
        except Exception:
            return np.array([])

def distribution_list():
    return {
        "Normal (Gaussian)": stats.norm,
        "Log-normal (lognorm)": stats.lognorm,
        "Gamma": stats.gamma,
        "Weibull (weibull_min)": stats.weibull_min,
        "Exponential": stats.expon,
        "Beta": stats.beta,
        "Chi-square": stats.chi2,
        "Uniform": stats.uniform,
        "Pareto": stats.pareto,
        "Fisk (log-logistic)": stats.fisk,
        "Gumbel (gumbel_r)": stats.gumbel_r,
        "Generalized Extreme Value (genextreme)": stats.genextreme,
    }

PARAM_DESCRIPTIONS = {
    "Normal (Gaussian)": {
        "loc": "Mean (center) of the normal distribution.",
        "scale": "Standard deviation (spread). Must be > 0.",
    },
    "Log-normal (lognorm)": {
        "s": "Shape (sigma) of the underlying normal on log-scale; larger -> heavier right tail.",
        "loc": "Location offset (adds to values).",
        "scale": "Scale (exp(mu) for underlying normal if loc=0).",
    },
    "Gamma": {
        "a": "Shape (k); controls skewness. Larger -> more symmetric.",
        "loc": "Location (shift).",
        "scale": "Scale (theta); larger -> wider distribution.",
    },
    "Weibull (weibull_min)": {
        "c": "Shape; c<1 heavy tail, c>1 more peaked.",
        "loc": "Location (shift).",
        "scale": "Scale (stretch).",
    },
    "Exponential": {
        "loc": "Location (shift).",
        "scale": "Scale = 1/lambda (mean). Must be > 0.",
    },
    "Beta": {
        "a": "Alpha: first shape parameter.",
        "b": "Beta: second shape parameter.",
        "loc": "Lower bound of support (actual support: loc .. loc+scale).",
        "scale": "Width of support (typically >0).",
    },
    "Chi-square": {
        "df": "Degrees of freedom (shape). Usually > 0.",
        "loc": "Location (shift).",
        "scale": "Scale (stretch).",
    },
    "Uniform": {
        "loc": "Lower bound (min).",
        "scale": "Width of interval (max = loc + scale). Must be > 0.",
    },
    "Pareto": {
        "b": "Shape parameter (tail heaviness).",
        "loc": "Location (shift).",
        "scale": "Scale parameter.",
    },
    "Fisk (log-logistic)": {
        "c": "Shape parameter (controls tail behavior).",
        "loc": "Location (shift).",
        "scale": "Scale (stretch).",
    },
    "Gumbel (gumbel_r)": {
        "loc": "Location (mode).",
        "scale": "Scale (stretch).",
    },
    "Generalized Extreme Value (genextreme)": {
        "c": "Shape (sign controls tail direction).",
        "loc": "Location (shift).",
        "scale": "Scale (stretch).",
    },
}

def get_param_help(dist_name, param_name):
    info = PARAM_DESCRIPTIONS.get(dist_name, {})
    if param_name in info:
        return info[param_name]
    if param_name == "loc":
        return "Location: shifts the distribution left/right."
    if param_name == "scale":
        return "Scale: stretches/squeezes the distribution (must be > 0)."
    return "Shape parameter: controls skewness/peakedness/tail behaviour."

def safe_fit(dist, data):
    return dist.fit(data)

def evaluate_pdf(dist, params, x):
    try:
        y = dist.pdf(x, *params)
        y = np.array(y, dtype=float)
        return y
    except Exception:
        return np.full_like(x, np.nan, dtype=float)

def ks_statistic(dist, params, data):
    try:
        res = stats.kstest(data, dist.cdf, args=params)
        return res.statistic, res.pvalue
    except Exception:
        return np.nan, np.nan

def rmse_hist_pdf(data, dist, params, bins=30):
    hist, edges = np.histogram(data, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    pdf_vals = evaluate_pdf(dist, params, centers)
    if np.any(~np.isfinite(pdf_vals)):
        return np.nan
    return np.sqrt(np.mean((hist - pdf_vals) ** 2))

def get_param_names(dist):
    shapes = dist.shapes
    if shapes:
        s = [p.strip() for p in shapes.split(",")]
    else:
        s = []
    s += ["loc", "scale"]
    return s

def create_slider_for_param(name, data_min, data_max, default):
    span = max(1.0, data_max - data_min)
    if name == "loc":
        low = data_min - span * 2
        high = data_max + span * 2
        step = max(0.001, span / 100.0)
    elif name == "scale":
        low = max(1e-6, span / 100.0)
        high = max(span * 5, 1.0)
        step = max(1e-6, high / 100.0)
    else:
        low = -10.0
        high = 10.0
        step = 0.01
    if default is not None and np.isfinite(default):
        if default < low:
            low = default * 1.5 if default < 0 else default - abs(default) * 0.5
        if default > high:
            high = default * 1.5 if default > 0 else default + abs(default) * 0.5
    return low, high, step

def rescale_to_01(data):
    amin = np.min(data)
    amax = np.max(data)
    if amax - amin <= 0:
        transformed = np.clip(data - amin, 0.0, 1.0)
        return transformed, (amin, amax)
    transformed = (data - amin) / (amax - amin)
    return transformed, (amin, amax)

def inverse_rescaled_pdf(pdf_vals, x_orig, amin, amax):
    jac = 1.0 / (amax - amin) if (amax - amin) != 0 else 1.0
    return pdf_vals * jac

def compute_log_likelihood(dist, params, data):
    pdf_vals = evaluate_pdf(dist, params, data)
    # Avoid log(0) by clipping to small positive
    eps = 1e-300
    pdf_vals = np.where(np.isfinite(pdf_vals), pdf_vals, 0.0)
    pdf_vals = np.clip(pdf_vals, eps, None)
    ll = np.sum(np.log(pdf_vals))
    return float(ll)

def aic_bic(ll, k, n):
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll
    return aic, bic

#illustrative plots for a parameter effect
def plot_param_effects(dist, param_name, base_params, param_idx, x_domain, is_rescaled=False, rescale_info=None):
  
    fig, ax = plt.subplots(figsize=(5, 2.5))
    x = np.linspace(x_domain[0], x_domain[1], 400)
    base = base_params[param_idx] if param_idx < len(base_params) else 1.0
    if param_name in ("scale", "s", "scale"):
        vals = [max(1e-6, base * 0.5), base, base * 2.0]
    else:
        vals = [base - max(0.5, abs(base) * 0.5), base, base + max(0.5, abs(base) * 0.5)]
    labels = ["low", "base", "high"]
    for v, lab, col in zip(vals, labels, ["C2", "C1", "C0"]):
        params_mod = list(base_params)
        # extend if missing
        while len(params_mod) < param_idx + 1:
            params_mod.append(0.0)
        params_mod[param_idx] = v
        y = evaluate_pdf(dist, tuple(params_mod), x)
        if is_rescaled and rescale_info is not None:
            amin, amax = rescale_info
            xt = (x - amin) / (amax - amin) if (amax - amin) != 0 else x
            y_t = evaluate_pdf(dist, tuple(params_mod), xt)
            y = inverse_rescaled_pdf(y_t, x, amin, amax)
        if np.any(np.isfinite(y)):
            ax.plot(x, np.where(np.isfinite(y), y, np.nan), color=col, label=f"{lab} ({v:.3g})")
    ax.set_title(f"Effect of {param_name}")
    ax.legend(fontsize="small", ncol=3)
    ax.set_xlabel("x")
    return fig


if "data" not in st.session_state:
    st.session_state["data"] = np.array([], dtype=float)
if "last_text" not in st.session_state:
    st.session_state["last_text"] = ""
if "last_upload_name" not in st.session_state:
    st.session_state["last_upload_name"] = ""
if "last_generated" not in st.session_state:
    st.session_state["last_generated"] = False


st.title("Histogram Generator")
st.markdown(
    "Enter numeric data manually, upload a CSV, or generate sample data. "
)

left_col, right_col = st.columns([1.3, 2.2])

with left_col:
    st.header("Data input")
    data_input_mode = st.radio("How would you provide data?", ("Manual entry", "Upload CSV", "Generate sample"), index=0)

    # Manual entry
    if data_input_mode == "Manual entry":
        text = st.text_area("Paste numbers separated by commas, spaces, or newlines", height=150, placeholder="e.g. 1.2, 3.4, 2.2, ...")
        if text != st.session_state.get("last_text", ""):
            parsed = parse_text_data(text)
            st.session_state["data"] = parsed
            st.session_state["last_text"] = text
        st.caption(f"Numbers parsed: {len(st.session_state['data'])}")

    elif data_input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV file (single column or arbitrary shape)", type=["csv", "txt"])
        if uploaded is not None:
            if uploaded.name != st.session_state.get("last_upload_name", ""):
                parsed = read_csv_file(uploaded)
                st.session_state["data"] = parsed
                st.session_state["last_upload_name"] = uploaded.name
            st.caption(f"Numbers read from file: {len(st.session_state['data'])}")
            if len(st.session_state['data']) == 0:
                st.error("No numeric data found in the uploaded file.")
    else:
        st.subheader("Generate sample data")
        sample_dist_name = st.selectbox("Choose a sampling distribution", list(distribution_list().keys()), index=0)
        sample_n = st.number_input("Sample size", min_value=10, max_value=200000, value=500, step=10)
        seed = st.number_input("RNG seed (0 = random)", value=0, step=1)
        rng = np.random.default_rng(int(seed)) if seed != 0 else np.random.default_rng()
        if st.button("Generate sample"):
            dist_obj = distribution_list()[sample_dist_name]
            try:
                if dist_obj is stats.norm:
                    data_gen = rng.normal(loc=0.0, scale=1.0, size=sample_n)
                elif dist_obj is stats.lognorm:
                    data_gen = stats.lognorm.rvs(1.0, loc=0, scale=1.0, size=sample_n, random_state=rng)
                elif dist_obj is stats.gamma:
                    data_gen = stats.gamma.rvs(2.0, loc=0, scale=2.0, size=sample_n, random_state=rng)
                elif dist_obj is stats.weibull_min:
                    data_gen = stats.weibull_min.rvs(1.5, loc=0, scale=1.0, size=sample_n, random_state=rng)
                elif dist_obj is stats.expon:
                    data_gen = stats.expon.rvs(loc=0, scale=1.0, size=sample_n, random_state=rng)
                elif dist_obj is stats.beta:
                    data_gen = stats.beta.rvs(2.0, 5.0, loc=0, scale=1.0, size=sample_n, random_state=rng)
                elif dist_obj is stats.chi2:
                    data_gen = stats.chi2.rvs(4.0, loc=0, scale=1.0, size=sample_n, random_state=rng)
                elif dist_obj is stats.uniform:
                    data_gen = stats.uniform.rvs(loc=-1.0, scale=2.0, size=sample_n, random_state=rng)
                elif dist_obj is stats.pareto:
                    data_gen = stats.pareto.rvs(3.0, size=sample_n, random_state=rng)
                elif dist_obj is stats.fisk:
                    data_gen = stats.fisk.rvs(3.0, loc=0, scale=1.0, size=sample_n, random_state=rng)
                elif dist_obj is stats.gumbel_r:
                    data_gen = stats.gumbel_r.rvs(loc=0, scale=1.0, size=sample_n, random_state=rng)
                else:
                    data_gen = rng.normal(size=sample_n)
                st.session_state["data"] = np.array(data_gen, dtype=float)
                st.session_state["last_generated"] = True
                st.success(f"Sample generated ({len(data_gen)} points)")
            except Exception as e:
                st.error(f"Error generating sample: {e}")

    st.markdown("---")
    st.header("Fitting options")
    all_dists = distribution_list()
    dist_choices = st.multiselect("Choose distributions to fit (you can select multiple)", options=list(all_dists.keys()), default=["Normal (Gaussian)"])
    bins = st.slider("Histogram bins", min_value=10, max_value=300, value=40)
    show_density = st.checkbox("Normalize histogram to density (recommended)", value=True)
    overlay_all = st.checkbox("Overlay all selected fits on histogram", value=True)

    st.markdown("Domain / Transform helpers")
    rescale_beta = st.checkbox("Rescale data to [0,1] before fitting Beta distribution (recommended)", value=True)
    auto_positive = st.checkbox("If a chosen distribution requires positive data (Gamma/Weibull/Exponential/Pareto), shift data to positive (add constant) where necessary", value=False)

    st.markdown("Export / Save")
    if st.session_state["data"].size > 0:
        df = pd.DataFrame({"value": st.session_state["data"]})
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download parsed data as CSV", data=csv_bytes, file_name="data.csv", mime="text/csv")

with right_col:
    st.header("Fit results & visualization")
    data = st.session_state["data"]
    if data is None or data.size == 0:
        st.info("Provide or generate data on the left to begin fitting.")
        st.stop()

    results = []
    n = len(data)
    for name in dist_choices:
        dist = all_dists[name]
        data_for_fit = np.array(data, dtype=float)
        transform_applied = None
        transform_info = None

        if name.startswith("Beta") and rescale_beta:
            data_for_fit, transform_info = rescale_to_01(data_for_fit)
            transform_applied = "rescale_01"

        if auto_positive and name in ("Gamma", "Weibull (weibull_min)", "Exponential", "Pareto", "Chi-square"):
            if np.min(data_for_fit) <= 0:
                shift = abs(np.min(data_for_fit)) + 1e-6
                data_for_fit = data_for_fit + shift
                transform_applied = "shift_positive"
                transform_info = shift

        try:
            params = safe_fit(dist, data_for_fit)
            ll = compute_log_likelihood(dist, params, data_for_fit)
            k = len(params)  # number of fitted parameters
            aic, bic = aic_bic(ll, k, len(data_for_fit))
            rmse = rmse_hist_pdf(data_for_fit, dist, params, bins=bins)
            ks_stat, ks_p = ks_statistic(dist, params, data_for_fit)
            results.append({
                "name": name,
                "dist": dist,
                "params": params,
                "rmse": float(rmse) if np.isfinite(rmse) else None,
                "ks_stat": float(ks_stat) if np.isfinite(ks_stat) else None,
                "ks_pvalue": float(ks_p) if np.isfinite(ks_p) else None,
                "loglik": float(ll),
                "aic": float(aic),
                "bic": float(bic),
                "transform": transform_applied,
                "transform_info": transform_info,
                "data_for_fit": data_for_fit,
            })
        except Exception as e:
            results.append({
                "name": name,
                "dist": dist,
                "params": None,
                "rmse": None,
                "ks_stat": None,
                "ks_pvalue": None,
                "loglik": None,
                "aic": None,
                "bic": None,
                "transform": None,
                "transform_info": None,
                "data_for_fit": None,
                "error": str(e)
            })

    if len(results) == 0:
        st.info("No distributions selected.")
        st.stop()

    #Summary table 
    summary_rows = []
    for r in results:
        summary_rows.append({
            "Distribution": r["name"],
            "RMSE": "" if r["rmse"] is None else f"{r['rmse']:.6g}",
            "AIC": "" if r.get("aic") is None else f"{r['aic']:.6g}",
            "BIC": "" if r.get("bic") is None else f"{r['bic']:.6g}",
            "KS p-value": "" if r.get("ks_pvalue") is None else f"{r['ks_pvalue']:.6g}",
            "Fit succeeded": ("Yes" if r.get("params") is not None else "No"),
            "Error": r.get("error", "")
        })
    st.subheader("Fit summary")
    st.table(pd.DataFrame(summary_rows))


    selected_for_detail = st.selectbox("Select a distribution for detailed view and manual fitting", options=[r["name"] for r in results], index=0)
    detail = next(r for r in results if r["name"] == selected_for_detail)

    st.subheader(f"Detailed view — {selected_for_detail}")

    if detail.get("params") is not None:
        param_tuple = detail["params"]
        param_names = get_param_names(detail["dist"])
        param_dict = {}
        for i, nm in enumerate(param_names):
            try:
                param_dict[nm] = float(param_tuple[i])
            except Exception:
                param_dict[nm] = None
        st.write("Fitted parameters (from scipy.stats.fit):")
        st.json(param_dict)
    else:
        st.warning("Fitting failed for this distribution.")
        param_names = get_param_names(detail["dist"])
        param_dict = {nm: None for nm in param_names}

    #manual fitting controls
    st.markdown("### Manual fitting")
    st.write("Toggle manual fitting to reveal sliders. Expand the examples to see visual effect.")
    manual_mode = st.checkbox("Enable manual fitting with sliders", value=False)

    data_min, data_max = np.min(data), np.max(data)
    defaults = {}
    if detail.get("params") is not None:
        for i, nm in enumerate(param_names):
            try:
                defaults[nm] = float(detail["params"][i])
            except Exception:
                defaults[nm] = 0.0
    else:
        for nm in param_names:
            if nm == "scale":
                defaults[nm] = (data_max - data_min) / 4.0 if (data_max - data_min) > 0 else 1.0
            elif nm == "loc":
                defaults[nm] = data_min
            else:
                defaults[nm] = 1.0

    manual_params = {}
    manual_examples = {}  

    if manual_mode:
        cols = st.columns(2)
        for i, nm in enumerate(param_names):
            default_val = defaults.get(nm, 0.0)
            low, high, step = create_slider_for_param(nm, data_min, data_max, default_val)
            with cols[i % 2]:
                try:
                    if abs(high - low) < 1e-12:
                        val = st.number_input(nm, value=float(default_val))
                    else:
                        val = st.slider(nm, min_value=float(low), max_value=float(high), value=float(default_val), step=float(step))
                except Exception:
                    val = st.number_input(nm, value=float(default_val))
                st.caption(get_param_help(selected_for_detail, nm))
                with st.expander("Show examples for " + nm, expanded=False):
                    base_params = list(param_tuple) if param_tuple is not None else [defaults.get(p, 1.0) for p in param_names]
                    #determine xdomain for examples in original data space
                    xpad = 0.05 * (data_max - data_min) if data_max > data_min else 1.0
                    x_domain = (data_min - xpad, data_max + xpad)
                    is_rescaled = (detail.get("transform") == "rescale_01")
                    rescale_info = detail.get("transform_info") if is_rescaled else None
                    fig = plot_param_effects(detail["dist"], nm, base_params, i, x_domain, is_rescaled=is_rescaled, rescale_info=rescale_info)
                    st.pyplot(fig)
                manual_params[nm] = float(val)
        params_to_plot = tuple(manual_params[nm] for nm in param_names)
    else:
        params_to_plot = detail.get("params")

    fig, ax = plt.subplots(figsize=(9, 5))
    density_flag = show_density
    ax.hist(data, bins=bins, density=density_flag, alpha=0.35, label="Data histogram")

    x_min = np.min(data)
    x_max = np.max(data)
    if x_max - x_min == 0:
        x = np.linspace(x_min - 1, x_min + 1, 400)
    else:
        x = np.linspace(x_min - 0.05*(x_max-x_min), x_max + 0.05*(x_max-x_min), 1000)

    for r in results:
        if r.get("params") is None:
            continue
        try:
            if r["transform"] == "rescale_01" and r["transform_info"] is not None:
                amin, amax = r["transform_info"]
                xt = (x - amin) / (amax - amin) if (amax - amin) != 0 else x
                y_t = evaluate_pdf(r["dist"], r["params"], xt)
                y = inverse_rescaled_pdf(y_t, x, amin, amax)
            elif r["transform"] == "shift_positive" and r["transform_info"] is not None:
                shift = r["transform_info"]
                x_shifted = x + shift
                y = evaluate_pdf(r["dist"], r["params"], x_shifted)
            else:
                y = evaluate_pdf(r["dist"], r["params"], x)
            #Plot only if some finite values exist
            if np.any(np.isfinite(y)):
                ax.plot(x, np.where(np.isfinite(y), y, np.nan), label=r["name"], linewidth=1.5, alpha=0.9)
        except Exception:
            continue

    if params_to_plot is not None:
        try:
            if detail.get("transform") == "rescale_01" and detail.get("transform_info") is not None:
                amin, amax = detail.get("transform_info")
                xt = (x - amin) / (amax - amin) if (amax - amin) != 0 else x
                y_t = evaluate_pdf(detail["dist"], params_to_plot, xt)
                y = inverse_rescaled_pdf(y_t, x, amin, amax)
            elif detail.get("transform") == "shift_positive" and detail.get("transform_info") is not None:
                shift = detail.get("transform_info")
                x_shifted = x + shift
                y = evaluate_pdf(detail["dist"], params_to_plot, x_shifted)
            else:
                y = evaluate_pdf(detail["dist"], params_to_plot, x)
            if np.any(np.isfinite(y)):
                label = f"{selected_for_detail} ({'manual' if manual_mode else 'fitted'})"
                ax.plot(x, np.where(np.isfinite(y), y, np.nan), label=label, color="black", linewidth=2.5)
            else:
                st.warning("The chosen parameters produce a non-finite PDF; adjust parameters (e.g., ensure scale > 0).")
        except Exception as e:
            st.error(f"Unable to plot PDF for the selected distribution: {e}")

    ax.set_xlabel("Value")
    ax.set_ylabel("Density" if density_flag else "Count")
    ax.legend()
    st.pyplot(fig)

    st.markdown("### Fit parameters & quality (per selected distribution)")
    if detail.get("params") is not None:
        st.write("Fitted parameters:")
        pretty = [(nm, float(detail["params"][i]) if i < len(detail["params"]) else None) for i, nm in enumerate(param_names)]
        st.table(pd.DataFrame(pretty, columns=["Parameter", "Value"]))

        st.write("Quality metrics (on the data used for fitting):")
        if detail.get("loglik") is not None:
            st.write(f"- Log-likelihood: {detail['loglik']:.6g}")
            st.write(f"- AIC: {detail['aic']:.6g}")
            st.write(f"- BIC: {detail['bic']:.6g}")
        else:
            st.write("- Log-likelihood / AIC / BIC: N/A")
        if detail.get("rmse") is not None:
            st.write(f"- RMSE (histogram vs PDF on fitted data): {detail['rmse']:.6g}")
        if detail.get("ks_stat") is not None:
            st.write(f"- KS statistic: {detail['ks_stat']:.6g}")
            st.write(f"- KS p-value: {detail['ks_pvalue']:.6g}")

    out_rows = []
    for r in results:
        out_rows.append({
            "distribution": r["name"],
            "params": ",".join([str(x) for x in r["params"]]) if r.get("params") is not None else "",
            "loglik": r.get("loglik"),
            "aic": r.get("aic"),
            "bic": r.get("bic"),
            "rmse": r.get("rmse"),
            "ks_pvalue": r.get("ks_pvalue"),
            "transform": r.get("transform")
        })
    out_df = pd.DataFrame(out_rows)
    csv_buf = out_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download fit summary CSV", data=csv_buf, file_name="fit_summary.csv", mime="text/csv")

st.markdown("---")
st.caption("Notes: scipy.stats.fit is used. For distributions needing a specific domain (e.g., Beta), an optional automatic rescale is available and the plotted PDF accounts for the transform (Jacobian). Manual fitting now updates the plot live and session state preserves generated / uploaded data so plots don't disappear when toggling controls.")