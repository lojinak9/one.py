

import streamlit as st
import numpy as np
import sympy as sp

st.set_page_config(page_title="Numerical Methods Solver", layout="wide")
st.title("🔢 Numerical Methods Project")

# القائمة الجانبية
method = st.sidebar.selectbox("Select Method", 
    ["Bisection", "False Position", "Simple Fixed Point", "Newton", "Secant", "Gauss Elimination", "LU Decomposition", "Cramer's Rule"])

# --- الجزء الخاص بحل الجذور (Roots) ---
if method in ["Bisection", "False Position", "Simple Fixed Point", "Newton", "Secant"]:
    st.header(f"Method: {method}")
    func_input = st.text_input("Enter Function f(x):", "x**2 - 4")
    eps = st.number_input("Enter Tolerance (eps %):", value=0.5)
    
    col1, col2 = st.columns(2)
    xl_input = col1.number_input("Lower Value (xl / x0):", value=0.0)
    xu_input = col2.number_input("Upper Value (xu / x1):", value=3.0)

    if st.button("Solve"):
        try:
            x = sp.symbols('x')
            f_sym = sp.sympify(func_input)
            f = sp.lambdify(x, f_sym)
            df = sp.lambdify(x, sp.diff(f_sym, x))
            
            rows = []
            iter_count = 0
            xr, error = 0, 100
            xl, xu = xl_input, xu_input

            if method == "Bisection":
                while error > eps and iter_count < 20:
                    xr_old = xr
                    xr = (xl + xu) / 2
                    if iter_count != 0: error = abs((xr - xr_old) / xr) * 100
                    # هذه هي التفاصيل التي يحتاجها الدكتور (نفس اللي في كودك)
                    rows.append([iter_count, xl, f(xl), xu, f(xu), xr, f(xr), f"{error:.4f}%"])
                    if f(xl) * f(xr) < 0: xu = xr
                    else: xl = xr
                    iter_count += 1
                st.table(st.dataframe(np.array(rows), column_config={
                    "0": "Iter", "1": "xl", "2": "f(xl)", "3": "xu", "4": "f(xu)", "5": "xr", "6": "f(xr)", "7": "Error%"}))

            elif method == "Newton":
                curr_x = xl_input
                while error > eps and iter_count < 20:
                    xi_next = curr_x - (f(curr_x) / df(curr_x))
                    error = abs((xi_next - curr_x) / xi_next) * 100
                    rows.append([iter_count, curr_x, f(curr_x), df(curr_x), f"{error:.4f}%"])
                    curr_x = xi_next
                    iter_count += 1
                st.table(st.dataframe(np.array(rows), column_config={
                    "0": "i", "1": "Xi", "2": "f(Xi)", "3": "f'(Xi)", "4": "Error%"}))
            
            st.success(f"Final Root: {xr if method == 'Bisection' else curr_x}")
        except Exception as e:
            st.error(f"Error: {e}")

# --- الجزء الخاص بالمصفوفات (Systems) ---
else:
    st.header(f"Method: {method}")
    st.write("Solving a 2x2 System: ax + by = c | dx + ey = f")
    c1, c2, c3 = st.columns(3)
    a = c1.number_input("a:", value=1.0)
    b = c2.number_input("b:", value=2.0)
    c = c3.number_input("c:", value=3.0)
    d = c1.number_input("d:", value=4.0)
    e = c2.number_input("e:", value=5.0)
    f = c3.number_input("f:", value=6.0)

    if st.button("Solve System"):
        if method == "Cramer's Rule":
            D = a * e - b * d
            Dx = c * e - b * f
            Dy = a * f - c * d
            st.write(f"D = {D}, Dx = {Dx}, Dy = {Dy}")
            if D != 0: st.success(f"x = {Dx/D}, y = {Dy/D}")
            else: st.error("No unique solution")
            
        elif method == "LU Decomposition":
            l21 = d / a
            u22 = e - l21 * b
            y2 = f - l21 * c
            y_val = y2 / u22
            x_val = (c - b * y_val) / a
            st.write(f"L Matrix: [[1, 0], [{l21}, 1]]")
            st.write(f"U Matrix: [[{a}, {b}], [0, {u22}]]")
            st.success(f"x = {x_val}, y = {y_val}")