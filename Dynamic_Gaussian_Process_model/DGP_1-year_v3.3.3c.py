import os
# Disable GPflow shape checks
os.environ['GPFLOW_CHECK_SHAPES'] = '0'
os.environ['CHECK_SHAPES'] = 'False'

import random
import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.set_visible_devices([], device_type='GPU')
from gpflow.optimizers import Scipy
import gpflow
from gpflow.utilities import parameter_dict, multiple_assign
from gpflow.mean_functions import Constant, Linear

from scipy.interpolate import CubicSpline

# Run 1-month ahead simulation to check the fitting of the model

# =============================================================================
# 1. Configuration
# =============================================================================
SEED = 513
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

LOAD_STATIC_PARAMS = True # if choose True, need to adjust import setting such as importing inducing points or not in Sec4.
SKIP_STATIC_CALIB = False

# File paths
WD = os.path.dirname(os.path.abspath(__file__)) + '/'
INPUT_YIELDS        = WD + 'daily_smith_wilson_20000101_.csv'
INPUT_PORTFOLIO     = WD + 'sample_portfolio_v3.1.csv'
INPUT_STATIC_PARAMS_FILE  = WD + 'static_calib_params_1-year_v3.3.3.npz'
OUTPUT_RISK         = WD + 'DGP_1-year_risk_v3.3.3c.csv'
SCENARIO_OUT        = WD + 'DGP_IR_scenarios_1-year_v3.3.3c.csv'
DYNAMIC_PARAMS_OUT  = WD + 'dgp_dynamic_params_1-year_v3.3.3c.npz'
OUT_STATIC_PARAMS_FILE  = WD + 'static_calib_params_1-year_v3.3.3c.npz'

# Inducing point counts
NUM_INDUCING_STATIC  = 400
NUM_INDUCING_DYNAMIC = 80

# Static initial training period
STATIC_START = '2000-01-01'
STATIC_END   = '2012-12-31'
# Dynamic backtest start (overlaps with static end)
DYNAMIC_START = '2006-01-01'

# Overall calibration end date
CAL_END = '2024-12-31'

# Backtest settings
WINDOW_SIZE      = 24   # months
FORECAST_HORIZON = 1    # months ahead
PRED_TIME        = 1.0 + FORECAST_HORIZON/(WINDOW_SIZE-1)

# Update frequencies
BATCH_UPDATE_FREQ   = 12   # windows
ONLINE_UPDATE_STEPS = 40   # Adam steps per window

# Scenario & risk
N_SIM           = 3000
QUANTILE_LEVEL  = 0.05
SCENARIO_SAVING = 12

# =============================================================================
# 2. Load & preprocess weekly yields
# =============================================================================
MATURITIES   = ['1','5','7','10','15','20','25','30','40']
MATURITIES_F = np.array([float(m) for m in MATURITIES])

df_raw = pd.read_csv(INPUT_YIELDS, parse_dates=['Date'])
for m in MATURITIES:
    df_raw[m] = pd.to_numeric(df_raw[m], errors='coerce') * 100.0

df_monthly = (
    df_raw.set_index('Date')
          .resample('M').last()
          .ffill()
)
dates = df_monthly.index

# index positions for static, dynamic, and calib end
static_start_idx  = dates.get_indexer([pd.to_datetime(STATIC_START)], method='bfill')[0]
static_end_idx    = dates.get_indexer([pd.to_datetime(STATIC_END)],   method='ffill')[0]
dynamic_start_idx = dates.get_indexer([pd.to_datetime(DYNAMIC_START)],method='bfill')[0]
calend_idx        = dates.get_indexer([pd.to_datetime(CAL_END)],      method='ffill')[0]

calib = df_monthly.iloc[static_start_idx:calend_idx+1]
dates = calib.index
NUM_WINDOWS = len(dates) - WINDOW_SIZE - FORECAST_HORIZON + 1

# long-format builder
def build_long_format(Y):
    X_list, Y_list = [], []
    win, tasks = Y.shape
    for t in range(win):
        time_norm = t/(win-1) if win>1 else 0.0
        for j in range(tasks):
            X_list.append([time_norm, MATURITIES_F[j], j])
            Y_list.append(Y[t,j])
    return np.array(X_list), np.array(Y_list).reshape(-1,1)

# =============================================================================
# 3. Kernel & Models
# =============================================================================
def make_kernel():
    k_time = (gpflow.kernels.RBF(1,active_dims=[0]) +
              gpflow.kernels.Linear(1,active_dims=[0]) +
              gpflow.kernels.RationalQuadratic(1,active_dims=[0]))
    k_mat  = (gpflow.kernels.RBF(1,active_dims=[1]) +
              gpflow.kernels.Linear(1,active_dims=[1]) +
              gpflow.kernels.Matern32(1,active_dims=[1]) +
              gpflow.kernels.RationalQuadratic(1,active_dims=[1]))
    coreg = gpflow.kernels.Coregion(output_dim=len(MATURITIES),rank=1,active_dims=[2])
    return (k_time + k_mat + k_time*k_mat) * coreg

kernel = make_kernel()
scipy_opt = Scipy()
adam      = tf.optimizers.Adam(1e-4)

# 3-A. Static model
mean_fn = Constant() + Linear(A=np.zeros((3,1)), b=np.zeros((1,)))

Zs = tf.convert_to_tensor(np.zeros((NUM_INDUCING_STATIC,3)),dtype=tf.float64)
static_model = gpflow.models.SVGP(kernel,
                                  gpflow.likelihoods.Gaussian(),
                                  Zs,
                                  mean_function=mean_fn,
                                  num_latent_gps=1,whiten=True)

# 3-B. Dynamic model placeholder (will assign params later)
Zd = tf.convert_to_tensor(np.zeros((NUM_INDUCING_DYNAMIC,3)),dtype=tf.float64)
dynamic_model = gpflow.models.SVGP(kernel,
                                   gpflow.likelihoods.Gaussian(),
                                   Zd,
                                   mean_function=mean_fn,
                                   num_latent_gps=1,whiten=True)

# =============================================================================
# 4. Static calibration
# =============================================================================
static_Y = calib.iloc[:static_end_idx-static_start_idx+1][MATURITIES].values
Xs, Ys    = build_long_format(static_Y)
Xs, Ys    = np.ascontiguousarray(Xs), np.ascontiguousarray(Ys)
Xs_tf     = tf.convert_to_tensor(Xs,dtype=tf.float64)
Ys_tf     = tf.convert_to_tensor(Ys,dtype=tf.float64)

perm = np.random.permutation(len(Xs))[:NUM_INDUCING_STATIC]
static_model.inducing_variable.Z.assign(Xs[perm])

if LOAD_STATIC_PARAMS and os.path.exists(INPUT_STATIC_PARAMS_FILE):
    data = np.load(INPUT_STATIC_PARAMS_FILE)
    load_exclude = ("inducing_variable", "q_mu", "q_sqrt")
    init_params = {
        k: data[k]
        for k in data.files
        if not any(substr in k for substr in load_exclude)
    }
    gpflow.utilities.multiple_assign(static_model, init_params)
    print(f'  >> Initialized static model from {INPUT_STATIC_PARAMS_FILE}')

if not SKIP_STATIC_CALIB:
    scipy_opt.minimize(
        static_model.training_loss_closure((Xs_tf, Ys_tf)),
        static_model.trainable_variables,
        options=dict(maxiter=800)
    )
    print('Static calibration done')

    static_params = gpflow.utilities.parameter_dict(static_model)
    save_exclude = ()
    to_save = {
        k: v for k, v in static_params.items()
        if not any(substr in k for substr in save_exclude)
    }
    np.savez(OUT_STATIC_PARAMS_FILE, **to_save)
    print(f'Saved static parameters to {OUT_STATIC_PARAMS_FILE}')
else:
    print('Static calibration skipped (SKIP_STATIC_CALIB=True)')

# extract static params (except Z)
static_params = parameter_dict(static_model)
exclude = ("inducing_variable", "q_mu", "q_sqrt")
filtered = {
    k: v
    for k, v in static_params.items()
    if not any(substr in k for substr in exclude)
}
multiple_assign(dynamic_model, filtered)

# =============================================================================
# 5. Load portfolio
# =============================================================================
port_df = pd.read_csv(INPUT_PORTFOLIO)
years   = 50
n_ports = port_df.filter(like='A').shape[1]
assets  = np.vstack([port_df[f'A{i+1}'].values[:years] for i in range(n_ports)])
liabs   = np.vstack([port_df[f'L{i+1}'].values[:years] for i in range(n_ports)])

# =============================================================================
# 6. Dynamic backtest
# =============================================================================
risk_results = []
all_scenarios = []
dynamic_start = (dynamic_start_idx - static_start_idx + 1) - WINDOW_SIZE + 1
if dynamic_start < 0: dynamic_start = 0

for w in range(dynamic_start, NUM_WINDOWS):
    start = dates[w]
    end   = dates[w+WINDOW_SIZE-1]
    test  = dates[w+WINDOW_SIZE]
    print(f"Window {w-dynamic_start+1}/{NUM_WINDOWS-dynamic_start}: {start.date()}~{end.date()} -> {test.date()}")

    Yw      = calib.loc[start:end][MATURITIES].values
    Xw, Yw_v = build_long_format(Yw)
    Xw_tf   = tf.convert_to_tensor(Xw,dtype=tf.float64)
    Yw_tf   = tf.convert_to_tensor(Yw_v,dtype=tf.float64)

    # initial dynamic inducing-point calibration
    if w == dynamic_start:
        scipy_opt.minimize(
            dynamic_model.training_loss_closure((Xw_tf,Yw_tf)),
            dynamic_model.trainable_variables,
            options=dict(maxiter=500)
        )
        print('  >> Initial dynamic inducing optimized')

    # periodic batch updates
    if (w-dynamic_start+1) % BATCH_UPDATE_FREQ == 0:
        scipy_opt.minimize(
            dynamic_model.training_loss_closure((Xw_tf,Yw_tf)),
            dynamic_model.trainable_variables,
            options=dict(maxiter=200)
        )
        print('  >> Periodic batch update')

    # online Adam updates
    for _ in range(ONLINE_UPDATE_STEPS):
        with tf.GradientTape() as tape:
            loss = dynamic_model.training_loss((Xw_tf,Yw_tf))
        grads = tape.gradient(loss, dynamic_model.trainable_variables)
        adam.apply_gradients(zip(grads,dynamic_model.trainable_variables))

    # hybrid L-BFGS refinement
    scipy_opt.minimize(
        dynamic_model.training_loss_closure((Xw_tf,Yw_tf)),
        dynamic_model.trainable_variables,
        options=dict(maxiter=20)
    )

    # prediction & scenario gen
    Xtest = np.array([[PRED_TIME,m,j] for j,m in enumerate(MATURITIES_F)],dtype=np.float64)
    fm, fc = dynamic_model.predict_f(
        tf.convert_to_tensor(Xtest,dtype=tf.float64), full_cov=True
    )
    fmean = fm.numpy().flatten()
    #cov_latent = fc.numpy()
    #noise_var  = dynamic_model.likelihood.variance.numpy()
    #cov2d = cov_latent + noise_var * np.eye(cov_latent.shape[-1])
    cov2d = fc.numpy()
    if cov2d.ndim==3: cov2d = cov2d[0]
    cov2d = 0.5*(cov2d+cov2d.T)
    eig, vec = np.linalg.eigh(cov2d)
    cov_psd = vec @ np.diag(np.clip(eig,1e-8,None)) @ vec.T
    sims = np.random.multivariate_normal(fmean,cov_psd,size=N_SIM)

    sims_cs = np.vstack([
        CubicSpline(MATURITIES_F,row,bc_type='natural')(np.arange(1,years+1))
        for row in sims
    ])/100.0
    mean_cs = CubicSpline(MATURITIES_F,fmean,bc_type='natural')(
        np.arange(1,years+1))/100.0
    sims_all = np.vstack([mean_cs,sims_cs])

    if (w-dynamic_start+1) % SCENARIO_SAVING == 0:
        df_snap = pd.DataFrame(sims_cs,columns=np.arange(1,years+1))
        df_snap.insert(0,'date',test)
        df_snap.insert(0,'sim_id', np.arange(1,N_SIM+1))
        all_scenarios.append(df_snap)

    total_pv = np.zeros((n_ports,N_SIM+1))
    t_vec    = np.arange(1,years+1)
    for s in range(N_SIM+1):
        dfs = 1.0/((1.0+sims_all[s])**t_vec)
        for p in range(n_ports):
            total_pv[p,s] = np.sum((assets[p]-liabs[p])*dfs)
    diffs    = total_pv[:,1:]-total_pv[:,:1]
    var_vals = -np.quantile(diffs,QUANTILE_LEVEL,axis=1)
    risk_results.append({
        'Test_Date':test,
        **{f'VaR_Portfolio{p}':var_vals[p] for p in range(n_ports)}
    })

# 7. Save outputs
pd.DataFrame(risk_results).to_csv(OUTPUT_RISK,index=False)
if all_scenarios:
    pd.concat(all_scenarios,ignore_index=True).to_csv(SCENARIO_OUT,index=False)
np.savez(DYNAMIC_PARAMS_OUT,**parameter_dict(dynamic_model))
print('Done.')
