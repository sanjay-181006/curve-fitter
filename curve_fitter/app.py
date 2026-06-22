import re
import numpy as np
from scipy.optimize import curve_fit
from scipy import special
import sympy as sp
from sklearn.linear_model import OrthogonalMatchingPursuit, LassoLarsIC, LassoCV
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, render_template
import warnings

warnings.filterwarnings("ignore")


app = Flask(__name__)

def normalize_function_names(text):
    replacements = {
        'arcsin': 'asin',
        'arccos': 'acos',
        'arctan': 'atan',
        'cosec': 'csc',
        'cosecant': 'csc',
        'secant': 'sec'
    }

    for old, new in replacements.items():
        text = re.sub(rf'\b{old}\b', new, text, flags=re.IGNORECASE)

    return text

def generate_compositional_library(X_matrix):
    """
    Constructs high-order polynomials, trig transformations, and composite functions.
    The dictionary is strictly ordered from computationally simple to complex.
    """
    num_samples, num_vars = X_matrix.shape
    library_features = []
    feature_names = []
    
    for i in range(num_vars):
        v = np.nan_to_num(X_matrix[:, i])
        v_name = "x" if num_vars == 1 else f"x{i+1}"
        abs_v = np.abs(v)
        
        raw_features = {
            f"{v_name}": v,
            f"{v_name}²": v**2,
            f"{v_name}³": v**3,
            f"sqrt(|{v_name}|)": np.sqrt(abs_v),
            f"log(1+|{v_name}|)": np.log1p(abs_v),
            f"sin({v_name})": np.sin(v),
            f"cos({v_name})": np.cos(v),
            f"tan({v_name})": np.tan(np.clip(v, -1.55, 1.55)),
            f"csc({v_name})": 1.0 / np.where(np.abs(np.sin(v)) < 1e-12, 1e-12, np.sin(v)),
            f"sec({v_name})": 1.0 / np.where(np.abs(np.cos(v)) < 1e-12, 1e-12, np.cos(v)),
            f"cot({v_name})": 1.0 / np.where(np.abs(np.tan(v)) < 1e-12, 1e-12, np.tan(v)),
            f"tanh({v_name})": np.tanh(v),
            f"asin({v_name})": np.arcsin(np.clip(v, -1, 1)),
            f"acos({v_name})": np.arccos(np.clip(v, -1, 1)),
            f"atan({v_name})": np.arctan(v),
            f"sinh({v_name})": np.sinh(np.clip(v, -5, 5)),
            f"cosh({v_name})": np.cosh(np.clip(v, -5, 5)),
            f"tanh⁻¹({v_name})": np.arctanh(np.clip(v, -0.999999, 0.999999)),
            f"sinh⁻¹({v_name})": np.arcsinh(v),
            f"cosh⁻¹({v_name})": np.arccosh(np.maximum(v, 1.0)),
            f"sin({v_name}²)": np.sin(v**2),
            f"abs(sin({v_name}²))": np.abs(np.sin(v**2)),
            f"exp(-{v_name}²)": np.exp(-np.clip(v**2, 0, 50)),
            f"exp({v_name})": np.exp(np.clip(v, -10, 10)),
            f"2^{v_name}": np.exp2(np.clip(v, -50, 50)),
            f"3^{v_name}": np.power(3.0, np.clip(v, -25, 25)),
            f"5^{v_name}": np.power(5.0, np.clip(v, -15, 15)),
            f"10^{v_name}": np.power(10.0, np.clip(v, -10, 10)),
            f"e^(-{v_name})": np.exp(np.clip(-v, -10, 10)),
            f"e^sqrt(|{v_name}|)": np.exp(np.clip(np.sqrt(abs_v), 0, 10)),
            f"e^({v_name}²)": np.exp(np.clip(v**2, 0, 10)),
            f"e^(-|{v_name}|)": np.exp(-np.clip(abs_v, 0, 50)),
            f"e^(-sqrt(|{v_name}|))": np.exp(-np.sqrt(abs_v)),
            f"{v_name}*exp({v_name})": v * np.exp(np.clip(v, -10, 10)),
            f"{v_name}²*exp({v_name})": (v**2) * np.exp(np.clip(v, -10, 10)),
            f"{v_name}³*exp({v_name})": (v**3) * np.exp(np.clip(v, -10, 10)),
            f"1/(1+{v_name}²)": 1.0 / (1.0 + v**2),
            f"{v_name}*sin({v_name})": v * np.sin(v),
            f"{v_name}*cos({v_name})": v * np.cos(v),
            f"1/(1+{v_name})": 1.0 / np.where(np.abs(1 + v) < 1e-12, 1e-12, 1 + v),
            f"1/{v_name}": 1.0 / np.where(np.abs(v) < 1e-12, 1e-12, v),
            f"1/{v_name}²": 1.0 / np.where(np.abs(v**2) < 1e-12, 1e-12, v**2),
            f"log(|{v_name}|+1e-6)": np.log(np.abs(v) + 1e-6),
            f"exp(-|{v_name}|)": np.exp(-np.abs(v)),
            # === Inserted special and advanced features ===
            f"log2(1+|{v_name}|)": np.log2(1 + abs_v),
            f"log10(1+|{v_name}|)": np.log10(1 + abs_v),
            f"exp2({v_name})": np.exp2(np.clip(v, -10, 10)),
            f"floor({v_name})": np.floor(v),
            f"ceil({v_name})": np.ceil(v),
            f"round({v_name})": np.round(v),
            f"frac({v_name})": v - np.floor(v),
            f"sinc({v_name})": np.sinc(v / np.pi),
            f"sigmoid({v_name})": 1.0 / (1.0 + np.exp(-np.clip(v, -50, 50))),
            f"gaussian({v_name})": np.exp(-np.clip(v**2, 0, 50)),
            f"erf({v_name})": special.erf(v),
            f"erfc({v_name})": special.erfc(v),
            f"gamma(|{v_name}|+1)": special.gamma(np.clip(abs_v + 1, 1e-6, 20)),
            f"gammaln(|{v_name}|+1)": special.gammaln(np.clip(abs_v + 1, 1e-6, 20)),
            f"bessel_j0({v_name})": special.j0(v),
            f"bessel_j1({v_name})": special.j1(v),
            f"bessel_y0(|{v_name}|+1e-6)": special.y0(abs_v + 1e-6),
            f"bessel_y1(|{v_name}|+1e-6)": special.y1(abs_v + 1e-6),
            f"i0({v_name})": special.i0(np.clip(v, -50, 50)),
            f"i1({v_name})": special.i1(np.clip(v, -50, 50)),
            f"hypot({v_name},1)": np.hypot(v, 1.0),
            f"sinh({v_name})/{v_name}": np.where(np.abs(v) < 1e-12, 1.0, np.sinh(v)/v),
            f"tanh({v_name})/{v_name}": np.where(np.abs(v) < 1e-12, 1.0, np.tanh(v)/v),
            f"sin({v_name})/{v_name}": np.where(np.abs(v) < 1e-12, 1.0, np.sin(v)/v),
            f"cos({v_name})/{v_name}": np.where(np.abs(v) < 1e-12, 1.0, np.cos(v)/v),
            f"sqrt(1+{v_name}²)": np.sqrt(1 + v**2),
            f"1/sqrt(1+{v_name}²)": 1.0 / np.sqrt(1 + v**2),
            f"atanh(tanh({v_name}))": np.arctanh(np.clip(np.tanh(v), -0.999999, 0.999999)),
            f"sin(exp({v_name}))": np.sin(np.exp(np.clip(v, -5, 5))),
            f"cos(exp({v_name}))": np.cos(np.exp(np.clip(v, -5, 5))),
            f"exp(sin({v_name}))": np.exp(np.sin(v)),
            f"exp(cos({v_name}))": np.exp(np.cos(v)),
            f"sin(log(1+|{v_name}|))": np.sin(np.log1p(abs_v)),
            f"cos(log(1+|{v_name}|))": np.cos(np.log1p(abs_v)),
            f"x*exp(-x²)".replace('x', v_name): v * np.exp(-np.clip(v**2, 0, 50)),
            f"x²*exp(-x²)".replace('x', v_name): (v**2) * np.exp(-np.clip(v**2, 0, 50)),
            f"x³*exp(-x²)".replace('x', v_name): (v**3) * np.exp(-np.clip(v**2, 0, 50)),
            f"sin({v_name})²": np.sin(v)**2,
            f"cos({v_name})²": np.cos(v)**2,
            f"tanh({v_name})²": np.tanh(v)**2,
            f"abs({v_name})": np.abs(v),
            # === End special and advanced features ===
            # === Inserted advanced feature set ===
            f"{v_name}^0.25": np.power(abs_v + 1e-12, 0.25),
            f"{v_name}^0.5": np.sqrt(abs_v),
            f"{v_name}^1.5": np.power(abs_v + 1e-12, 1.5),
            f"{v_name}^2.5": np.power(abs_v + 1e-12, 2.5),
            f"{v_name}^4": v**4,
            f"{v_name}^5": v**5,
            f"{v_name}^6": v**6,
            f"({v_name}+1)^2": (v + 1)**2,
            f"({v_name}-1)^2": (v - 1)**2,

            f"2^(sin({v_name}))": np.power(2.0, np.sin(v)),
            f"2^(cos({v_name}))": np.power(2.0, np.cos(v)),
            f"e^(sin({v_name}))": np.exp(np.sin(v)),
            f"e^(cos({v_name}))": np.exp(np.cos(v)),
            f"e^({v_name}^3)": np.exp(np.clip(v**3, -10, 10)),
            f"e^(-{v_name}^3)": np.exp(np.clip(-(v**3), -10, 10)),

            f"log(|{v_name}|+1)": np.log(abs_v + 1.0),
            f"log2(|{v_name}|+1)": np.log2(abs_v + 1.0),
            f"log10(|{v_name}|+1)": np.log10(abs_v + 1.0),
            f"sqrt(log(|{v_name}|+2))": np.sqrt(np.log(abs_v + 2.0)),

            f"sin(2*{v_name})": np.sin(2*v),
            f"sin(3*{v_name})": np.sin(3*v),
            f"sin(5*{v_name})": np.sin(5*v),
            f"cos(2*{v_name})": np.cos(2*v),
            f"cos(3*{v_name})": np.cos(3*v),
            f"cos(5*{v_name})": np.cos(5*v),
            f"tan(2*{v_name})": np.tan(np.clip(2*v, -1.55, 1.55)),

            f"sin({v_name}^2)": np.sin(v**2),
            f"sin({v_name}^3)": np.sin(v**3),
            f"cos({v_name}^2)": np.cos(v**2),
            f"cos({v_name}^3)": np.cos(v**3),
            f"sin(sqrt(|{v_name}|))": np.sin(np.sqrt(abs_v)),
            f"cos(sqrt(|{v_name}|))": np.cos(np.sqrt(abs_v)),
            f"sin(log(1+|{v_name}|))": np.sin(np.log1p(abs_v)),
            f"cos(log(1+|{v_name}|))": np.cos(np.log1p(abs_v)),

            f"sinh(2*{v_name})": np.sinh(np.clip(2*v, -5, 5)),
            f"cosh(2*{v_name})": np.cosh(np.clip(2*v, -5, 5)),
            f"tanh(2*{v_name})": np.tanh(2*v),

            f"(sin({v_name}))^2": np.sin(v)**2,
            f"(sin({v_name}))^3": np.sin(v)**3,
            f"(cos({v_name}))^2": np.cos(v)**2,
            f"(cos({v_name}))^3": np.cos(v)**3,
            f"(sin({v_name})*cos({v_name}))": np.sin(v)*np.cos(v),
            f"{v_name}*sin({v_name}^2)": v*np.sin(v**2),
            f"{v_name}*cos({v_name}^2)": v*np.cos(v**2),
            f"{v_name}^2*sin({v_name})": (v**2)*np.sin(v),
            f"{v_name}^2*cos({v_name})": (v**2)*np.cos(v),

            f"exp(sin({v_name}^2))": np.exp(np.sin(v**2)),
            f"exp(cos({v_name}^2))": np.exp(np.cos(v**2)),
            f"sin(exp(sin({v_name})))": np.sin(np.exp(np.sin(v))),
            f"cos(exp(cos({v_name})))": np.cos(np.exp(np.cos(v))),
            f"log(1+sin({v_name})^2)": np.log1p(np.sin(v)**2),
            f"log(1+cos({v_name})^2)": np.log1p(np.cos(v)**2),

            f"1/(1+{v_name}^4)": 1.0/(1.0 + v**4),
            f"1/sqrt(1+{v_name}^4)": 1.0/np.sqrt(1.0 + v**4),
            f"gaussian2({v_name})": np.exp(-np.clip(v**4, 0, 50)),
            f"sech({v_name})": 1.0/np.cosh(np.clip(v, -5, 5)),
            f"csch({v_name})": 1.0/np.where(np.abs(np.sinh(v)) < 1e-12, 1e-12, np.sinh(v)),
            f"coth({v_name})": 1.0/np.where(np.abs(np.tanh(v)) < 1e-12, 1e-12, np.tanh(v)),
            # === End inserted advanced feature set ===
            f"sgn({v_name})": np.sign(v)
        }
        
        for name, arr in raw_features.items():
            library_features.append(arr)
            feature_names.append(name)

        base_items = list(raw_features.items())

        for name_a, arr_a in base_items[:20]:
            for name_b, arr_b in base_items[:20]:
                library_features.append(arr_a * arr_b)
                feature_names.append(f"({name_a})*({name_b})")

                library_features.append(arr_a + arr_b)
                feature_names.append(f"({name_a})+({name_b})")

    return np.column_stack(library_features), feature_names


def polynomial_exact_detector(X_data, target, max_degree=6, tol=1e-10):
    if X_data.shape[1] != 1:
        return None

    x = X_data[:, 0]

    for degree in range(1, min(max_degree, len(x) - 1) + 1):
        coeffs = np.polyfit(x, target, degree)
        pred = np.polyval(coeffs, x)

        if np.max(np.abs(pred - target)) < tol:
            return degree, coeffs

    return None


def linear_exact_detector(X_data, target, tol=1e-12):
    if X_data.shape[1] != 1:
        return None

    x = X_data[:, 0]

    if len(x) < 2:
        return None

    dx = np.diff(x)
    dy = np.diff(target)

    valid = np.abs(dx) > tol
    if not np.any(valid):
        return None

    slopes = dy[valid] / dx[valid]

    if np.max(np.abs(slopes - slopes[0])) < tol:
        m = slopes[0]
        b = target[0] - m * x[0]
        return m, b

    return None

def sinusoidal_exact_detector(X_data, target, tol=1e-6):
    if X_data.shape[1] != 1:
        return None

    x = X_data[:, 0]

    def sin_model(x, A, B, C, D):
        return A * np.sin(B * x + C) + D

    candidate_frequencies = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

    best = None
    best_error = np.inf

    for freq in candidate_frequencies:
        try:
            guess = [
                (np.max(target) - np.min(target)) / 2,
                freq,
                0.0,
                np.mean(target)
            ]

            params, _ = curve_fit(
                sin_model,
                x,
                target,
                p0=guess,
                maxfev=20000
            )

            pred = sin_model(x, *params)
            err = np.max(np.abs(pred - target))

            if err < best_error:
                best_error = err
                best = {
                    'params': params,
                    'predictions': pred,
                    'error': err
                }

        except Exception:
            pass

    if best is not None and best['error'] < tol:
        return best

    return None


def rational_exact_detector(X_data, target, tol=1e-8):
    if X_data.shape[1] != 1:
        return None

    x = X_data[:, 0]

    def reciprocal_model(x, a, b, c):
        denom = b * x + c
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        return a / denom

    guesses = [
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [target[0], 1.0, 1.0]
    ]

    best = None
    best_error = np.inf

    for guess in guesses:
        try:
            params, _ = curve_fit(
                reciprocal_model,
                x,
                target,
                p0=guess,
                maxfev=50000
            )

            pred = reciprocal_model(x, *params)
            err = np.max(np.abs(pred - target))

            if err < best_error:
                best_error = err
                best = (params, pred)
        except Exception:
            pass

    if best is not None and best_error < tol:
        return {
            'params': best[0],
            'predictions': best[1]
        }

    return None

def exponential_exact_detector(X_data, target, tol=1e-8):
    if X_data.shape[1] != 1:
        return None

    x = X_data[:, 0]

    if np.any(np.abs(target) < 1e-12):
        return None

    sign = np.sign(target[0])
    if not np.all(np.sign(target) == sign):
        return None

    y_abs = np.abs(target)

    try:
        coeffs = np.polyfit(x, np.log(y_abs), 1)
        lnB = coeffs[0]
        lnA = coeffs[1]

        A = np.exp(lnA) * sign
        B = np.exp(lnB)

        pred = A * (B ** x)

        if np.max(np.abs(pred - target)) < tol:
            return {
                'A': A,
                'B': B,
                'predictions': pred
            }
    except Exception:
        pass

    return None

def sign_exact_detector(X_data, target, tol=1e-8):
    if X_data.shape[1] != 1:
        return None

    x = X_data[:, 0]

    pred = np.sign(x)

    if np.max(np.abs(pred - target)) < tol:
        return {
            'predictions': pred,
            'model_name': 'y = sgn(x)'
        }

    return None


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fit', methods=['POST'])
def fit_curve():
    data = request.json
    if isinstance(data, dict):
        data = {
            k: normalize_function_names(v) if isinstance(v, str) else v
            for k, v in data.items()
        }
    try:
        points = np.array(data['points'], dtype=float)
    except (ValueError, KeyError, TypeError):
        return jsonify({'error': 'Malformed matrix layout received.'}), 400
        
    if points.ndim != 2:
        return jsonify({'error': 'Input matrix must be two-dimensional.'}), 400

    valid_mask = np.all(np.isfinite(points), axis=1)
    points = points[valid_mask]
    
    if len(points) < 3:
        return jsonify({'error': 'Insufficient dataset scope. Provide at least 3 valid coordinate lines.'}), 400
        
    num_cols = points.shape[1]
    
    try:
        if num_cols == 1:
            X_data = np.arange(1, len(points) + 1).reshape(-1, 1)
            target = points[:, 0]
            plot_dimensions = 2
        else:
            X_data = points[:, :-1]
            target = points[:, -1]
            plot_dimensions = num_cols

        X_features, feature_names = generate_compositional_library(X_data)
        

        sign_result = sign_exact_detector(X_data, target)
        if sign_result is not None:
            xi = np.linspace(min(X_data[:, 0]), max(X_data[:, 0]), 300)
            yi = np.sign(xi)

            return jsonify({
                'dimensions': int(plot_dimensions),
                'model_name': 'y = sign(x)',
                'r2_train': 1.0,
                'r2_val': 1.0,
                'actual': target.tolist(),
                'predicted': sign_result['predictions'].tolist(),
                'plot': {
                    'fit_x': xi.tolist(),
                    'fit_y': yi.tolist(),
                    'raw_x': X_data[:, 0].tolist()
                }
            })

        linear_result = linear_exact_detector(X_data, target)
        if linear_result is not None:
            m, b = linear_result

            predictions = m * X_data[:, 0] + b

            model_name = f"y = {m:.10g}*x + {b:.10g}"

            output_payload = {
                'dimensions': int(plot_dimensions),
                'model_name': model_name,
                'r2_train': 1.0,
                'r2_val': 1.0,
                'actual': target.tolist(),
                'predicted': predictions.tolist()
            }

            xi = np.linspace(min(X_data[:, 0]), max(X_data[:, 0]), 300)
            yi = m * xi + b

            output_payload['plot'] = {
                'fit_x': xi.tolist(),
                'fit_y': yi.tolist(),
                'raw_x': X_data[:, 0].tolist()
            }

            return jsonify(output_payload)

        # (Removed duplicate sign detector block)

        exponential_result = exponential_exact_detector(X_data, target)
        if exponential_result is not None:
            A = exponential_result['A']
            B = exponential_result['B']

            xi = np.linspace(min(X_data[:, 0]), max(X_data[:, 0]), 300)
            yi = A * (B ** xi)

            return jsonify({
                'dimensions': int(plot_dimensions),
                'model_name': f"y = {A:.10g}*({B:.10g})^x",
                'r2_train': 1.0,
                'r2_val': 1.0,
                'actual': target.tolist(),
                'predicted': exponential_result['predictions'].tolist(),
                'plot': {
                    'fit_x': xi.tolist(),
                    'fit_y': yi.tolist(),
                    'raw_x': X_data[:, 0].tolist()
                }
            })

        rational_result = rational_exact_detector(X_data, target)
        if rational_result is not None:
            a, b, c = rational_result['params']

            predictions = rational_result['predictions']

            model_name = f"y = ({a:.10g})/({b:.10g}*x + {c:.10g})"

            xi = np.linspace(min(X_data[:, 0]), max(X_data[:, 0]), 300)
            yi = a / (b * xi + c)

            return jsonify({
                'dimensions': int(plot_dimensions),
                'model_name': model_name,
                'r2_train': 1.0,
                'r2_val': 1.0,
                'actual': target.tolist(),
                'predicted': predictions.tolist(),
                'plot': {
                    'fit_x': xi.tolist(),
                    'fit_y': yi.tolist(),
                    'raw_x': X_data[:, 0].tolist()
                }
            })

        sinusoid_result = sinusoidal_exact_detector(X_data, target)
        if sinusoid_result is not None:
            A, B, C, D = sinusoid_result['params']

            x_symbol = sp.Symbol('x')
            expr = sp.simplify(A * sp.sin(B * x_symbol + C) + D)

            model_name = f"y = {sp.sstr(expr)}"

            xi = np.linspace(min(X_data[:, 0]), max(X_data[:, 0]), 300)
            yi = A * np.sin(B * xi + C) + D

            return jsonify({
                'dimensions': int(plot_dimensions),
                'model_name': model_name,
                'r2_train': 1.0,
                'r2_val': 1.0,
                'actual': target.tolist(),
                'predicted': sinusoid_result['predictions'].tolist(),
                'plot': {
                    'fit_x': xi.tolist(),
                    'fit_y': yi.tolist(),
                    'raw_x': X_data[:, 0].tolist()
                }
            })

        poly_result = None
        if len(points) >= 6:
            poly_result = polynomial_exact_detector(X_data, target)
        if poly_result is not None:
            degree, coeffs = poly_result

            terms = []
            power = degree

            for c in coeffs:
                if abs(c) < 1e-12:
                    power -= 1
                    continue

                if power == 0:
                    terms.append(f"{c:.10g}")
                elif power == 1:
                    terms.append(f"{c:.10g}*x")
                else:
                    terms.append(f"{c:.10g}*x^{power}")

                power -= 1

            predictions = np.polyval(coeffs, X_data[:, 0])

            model_name = "y = " + " + ".join(terms)

            output_payload = {
                'dimensions': int(plot_dimensions),
                'model_name': model_name,
                'r2_train': 1.0,
                'r2_val': 1.0,
                'actual': target.tolist(),
                'predicted': predictions.tolist()
            }

            xi = np.linspace(min(X_data[:, 0]), max(X_data[:, 0]), 300)
            yi = np.polyval(coeffs, xi)

            output_payload['plot'] = {
                'fit_x': xi.tolist(),
                'fit_y': yi.tolist(),
                'raw_x': X_data[:, 0].tolist()
            }

            return jsonify(output_payload)

        if X_data.shape[1] == 2:
            x1 = X_data[:, 0]
            x2 = X_data[:, 1]

            interaction_features = [
                x1 * x2,
                (x1 ** 2) * x2,
                x1 * (x2 ** 2)
            ]

            interaction_names = [
                'x1*x2',
                'x1²*x2',
                'x1*x2²'
            ]

            X_features = np.column_stack([
                X_features,
                *interaction_features
            ])

            feature_names.extend(interaction_names)

        feature_count = X_features.shape[1]
        feature_variance = np.var(X_features, axis=0)
        keep_mask = feature_variance > 1e-12
        X_features = X_features[:, keep_mask]
        feature_names = [n for n, k in zip(feature_names, keep_mask) if k]
        feature_count = X_features.shape[1]

        if len(points) <= 12:
            model = OrthogonalMatchingPursuit(
                n_nonzero_coefs=min(4, feature_count),
                fit_intercept=True
            )
            model.fit(X_features, target)
        else:
            try:
                model = LassoLarsIC(criterion='bic', fit_intercept=True)
                model.fit(X_features, target)
            except Exception:
                model = LassoCV(cv=min(5, len(points)), random_state=42)
                model.fit(X_features, target)

        coefs = model.coef_
        intercept = model.intercept_

        threshold = max(1e-6, 0.005 * np.max(np.abs(coefs)) if len(coefs) else 1e-6)
        significant_indices = [idx for idx, val in enumerate(coefs) if abs(val) > threshold]
        
        equation_terms = []

        for idx in significant_indices:
            weight = float(coefs[idx])
            name = feature_names[idx]

            coeff_str = f"{abs(weight):.6g}"

            if name == 'x':
                term = "x" if abs(weight - 1) < 1e-10 else f"{coeff_str}*x"
            else:
                if abs(weight - 1) < 1e-10:
                    term = name
                else:
                    term = f"{coeff_str}*{name}"

            if len(equation_terms) == 0:
                equation_terms.append(term if weight >= 0 else f"-{term}")
            else:
                equation_terms.append(
                    f" + {term}" if weight >= 0 else f" - {term}"
                )

        if abs(intercept) > threshold:
            intercept_str = f"{abs(float(intercept)):.6g}"

            if len(equation_terms) == 0:
                equation_terms.append(
                    intercept_str if intercept >= 0 else f"-{intercept_str}"
                )
            else:
                equation_terms.append(
                    f" + {intercept_str}"
                    if intercept >= 0
                    else f" - {intercept_str}"
                )

        model_name = "y = " + ("".join(equation_terms) if equation_terms else "0")

        full_predictions = np.asarray(model.predict(X_features), dtype=float)

        full_predictions = np.nan_to_num(full_predictions, nan=0.0).astype(float)
        target_clean = np.nan_to_num(target, nan=0.0).astype(float)
        
        ss_res = np.sum((target_clean - full_predictions) ** 2)
        ss_tot = np.sum((target_clean - np.mean(target_clean)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
        r2_score = float(r2_score)
        max_error = float(np.max(np.abs(target_clean - full_predictions)))

        if max_error < 1e-8 and (not significant_indices):
            model_name = "Exact relation detected, but symbolic family not yet implemented"

        output_payload = {
            'dimensions': int(plot_dimensions),
            'model_name': model_name,
            'r2_train': round(r2_score, 4),
            'r2_val': None,
            'actual': target_clean.tolist(),
            'predicted': full_predictions.tolist()
        }

        if plot_dimensions == 2:
            xi = np.linspace(min(X_data[:, 0]), max(X_data[:, 0]), 300).reshape(-1, 1)
            xi_features, _ = generate_compositional_library(xi)
            xi_features = xi_features[:, keep_mask]
            yi = np.asarray(model.predict(xi_features), dtype=float)
            yi = np.nan_to_num(yi, nan=0.0).astype(float)
            
            output_payload['plot'] = {
                'fit_x': xi.ravel().tolist(), 
                'fit_y': yi.tolist(), 
                'raw_x': X_data[:, 0].tolist()
            }
            
        elif plot_dimensions == 3:
            xi = np.linspace(min(X_data[:, 0]), max(X_data[:, 0]), 40)
            yi = np.linspace(min(X_data[:, 1]), max(X_data[:, 1]), 40)
            X_grid, Y_grid = np.meshgrid(xi, yi)
            grid_combined = np.column_stack([X_grid.flatten(), Y_grid.flatten()])
            grid_features, _ = generate_compositional_library(grid_combined)

            x1g = grid_combined[:, 0]
            x2g = grid_combined[:, 1]
            grid_features = np.column_stack([
                grid_features,
                x1g * x2g,
                (x1g ** 2) * x2g,
                x1g * (x2g ** 2)
            ])

            grid_features = grid_features[:, keep_mask]
            
            Z_grid = np.asarray(model.predict(grid_features), dtype=float)
            Z_grid = np.nan_to_num(Z_grid, nan=0.0).reshape(X_grid.shape).astype(float)
            
            output_payload['plot'] = {
                'fit_x': X_grid.tolist(), 
                'fit_y': Y_grid.tolist(), 
                'fit_z': Z_grid.tolist(),
                'raw_x': X_data[:, 0].tolist(), 
                'raw_y': X_data[:, 1].tolist()
            }

        if output_payload['r2_val'] is None:
            try:
                if len(target_clean) >= 10:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_features,
                        target_clean,
                        test_size=0.2,
                        random_state=42
                    )

                    if len(y_train) >= 3 and len(y_test) >= 2:
                        if len(points) <= 12:
                            val_model = OrthogonalMatchingPursuit(
                                n_nonzero_coefs=min(4, X_train.shape[1]),
                                fit_intercept=True
                            )
                        else:
                            try:
                                val_model = LassoLarsIC(criterion='bic', fit_intercept=True)
                            except Exception:
                                val_model = LassoCV(cv=min(5, len(y_train)), random_state=42)

                        val_model.fit(X_train, y_train)
                        y_pred_test = np.asarray(val_model.predict(X_test), dtype=float)

                        ss_res_val = np.sum((y_test - y_pred_test) ** 2)
                        ss_tot_val = np.sum((y_test - np.mean(y_test)) ** 2)

                        val_r2 = 1 - (ss_res_val / ss_tot_val) if ss_tot_val > 0 else 1.0
                        output_payload['r2_val'] = round(float(val_r2), 4)
                    else:
                        output_payload['r2_val'] = None
                else:
                    output_payload['r2_val'] = None
            except Exception:
                output_payload['r2_val'] = None

        return jsonify(output_payload)
        
    except Exception as e:
        return jsonify({'error': f"Mathematical execution failure: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)