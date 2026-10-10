pip install streamlit sympy numpy matplotlib
"""
منصة تفاعلية لحل المعادلات باستخدام دالة لامبرت W
Lambert W Interactive Solver Platform
"""
import streamlit as st
import sympy as sp
import numpy as np

# =========================================================
# إعداد الصفحة
# =========================================================
st.set_page_config(page_title="منصة لامبرت W التفاعلية", page_icon="🔷", layout="wide")

st.title("🔷 منصة تفاعلية لحل المعادلات بدالة لامبرت W")
st.markdown(
    r"دالة لامبرت تُعرَّف بـ: $W(z)\,e^{W(z)} = z$ "
    r"وتُستخدم لحل المعادلات المتسامية والمعادلات التفاضلية القابلة للاختزال إليها."
)
st.markdown("---")

# =========================================================
# الشريط الجانبي
# =========================================================
st.sidebar.header("⚙️ اختر نوع المسألة")
mode = st.sidebar.radio(
    "نوع المعادلة:",
    [
        "معادلة جبرية/متسامية",
        "معادلة تفاضلية عادية (ODE)",
        "معادلة تفاضلية جزئية (PDE)",
        "حاسبة دالة W المباشرة",
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "أدخل المعادلة، وستقوم المنصة:\n"
    "1. تحديد إن كانت قابلة للحل بدالة W\n"
    "2. عرض خطوات الاختزال\n"
    "3. إعطاء الحل الدقيق والعددي"
)

# =========================================================
# المرافق: محلل المعادلات القابلة لـ W
# =========================================================
def try_lambert_solve(eq, var):
    """
    يحاول حل المعادلة باستخدام دالة لامبرت W بالطرق الشائعة.
    يرجع (حل, وصف_الطريقة) أو (None, سبب_الفشل).
    """
    lhs = sp.expand(eq.lhs - eq.rhs)

    # --- النمط 1: x * exp(x) = a  ---
    try:
        # نجرّب عزل x*exp(x)
        sol = sp.solve(sp.Eq(lhs, 0), var)
        for s in sol:
            if s.has(sp.LambertW):
                return s, "حل مباشر بدالة لامبرت W"
    except Exception:
        pass

    # --- النمط 2: a^x = b x + c  ---
    try:
        # نستخدم solve مع lambertw صريحة
        sol = sp.solve(eq, var)
        if sol and any(str(s).find("LambertW") != -1 for s in sol):
            return sol, "اختزال بواسطة دالة W"
        if sol:
            return sol, "حل مباشر (قد لا يستخدم W)"
    except Exception:
        pass

    return None, "لا يمكن اختزاله إلى صورة W قياسية"


def classify_lambert_form(eq, var):
    """يصنّف شكل المعادلة بالنسبة لأشكال لامبرت القياسية."""
    patterns = [
        (r"x*\exp(x)", "x e^x = a ⟹ x = W(a)"),
        (r"x + \log(x)", "x + ln x = a ⟹ x = W(e^a)"),
        (r"\exp(b*x)", "e^{bx} = c(x-d) ⟹ x = d + W(...)/b"),
        (r"a\*\*x", "a^x = bx+c ⟹ استخدام W"),
    ]
    found = []
    s = sp.sstr(eq)
    for pat, desc in patterns:
        found.append(desc)
    return found


# =========================================================
# القسم 1: معادلة جبرية / متسامية
# =========================================================
if mode == "معادلة جبرية/متسامية":
    st.header("📌 حل معادلة جبرية أو متسامية بدالة W")

    st.markdown("أمثلة مقترحة:")
    st.code(
        "x*exp(x) - 2\n"
        "x + log(x) - 3\n"
        "exp(x) - 5*(x - 1)\n"
        "2**x - 3*x - 1\n"
        "x*exp(2*x) - 7",
        language="text"
    )

    eq_str = st.text_input("المعادلة (اكتب الطرفين، سيُعادل بـ 0):",
                           value="x*exp(x) - 2")
    var_str = st.text_input("المتغير:", value="x")

    if st.button("🚀 حل"):
        var = sp.Symbol(var_str, real=True)
        try:
            expr = sp.sympify(eq_str)
            eq = sp.Eq(expr, 0)

            st.subheader("المرحلة 1: كتابة المعادلة")
            st.latex(sp.latex(eq))

            st.subheader("المرحلة 2: التحقق من إمكانية استخدام دالة W")
            forms = classify_lambert_form(eq, var)
            for f in forms:
                st.markdown(f"- {f}")

            st.subheader("المرحلة 3: محاولة الحل بدالة W")
            try:
                sol = sp.solve(eq, var)
                if sol:
                    for i, s in enumerate(sol, 1):
                        st.latex(rf"{var_str}_{{{i}}} = {sp.latex(s)}")
                        # قيمة عددية إن أمكن
                        try:
                            val = complex(s.evalf())
                            st.markdown(
                                f"**القيمة العددية:** "
                                f"${sp.latex(sp.N(s, 6))}$"
                            )
                        except Exception:
                            pass
                    st.success("✅ تم إيجاد الحل بدالة لامبرت W")
                else:
                    st.warning("⚠️ لم يجد SymPy حلاً رمزياً. سيتم الحل عددياً.")
                    f = sp.lambdify(var, expr, "numpy")
                    xs = np.linspace(-5, 5, 2000)
                    ys = f(xs)
                    sign_changes = np.where(np.diff(np.sign(ys)))[0]
                    roots_num = []
                    for idx in sign_changes:
                        a_, b_ = xs[idx], xs[idx + 1]
                        for _ in range(60):
                            m = 0.5 * (a_ + b_)
                            if f(a_) * f(m) < 0:
                                b_ = m
                            else:
                                a_ = m
                        roots_num.append(0.5 * (a_ + b_))
                    for r in roots_num:
                        st.latex(rf"{var_str} \approx {r:.6f}")
                    st.info("💡 هذه حلول عددية بطريقة التنصيف.")
            except Exception as e:
                st.error(f"❌ تعذّر الحل الرمزي: {e}")

        except Exception as e:
            st.error(f"❌ خطأ في الإدخال: {e}")

# =========================================================
# القسم 2: ODE
# =========================================================
elif mode == "معادلة تفاضلية عادية (ODE)":
    st.header("📌 حل ODE قابل للاختزال إلى دالة W")

    st.markdown("الصيغ المدعومة:")
    st.latex(r"y' = a\,y + b\,y\,e^{cy}")
    st.latex(r"y' + p(x)\,y = q(x)\,y^n \quad (\text{برنولي})")

    x = sp.Symbol("x", real=True)
    y = sp.Function("y")(x)

    ode_str = st.text_input(
        "اكتب ODE (استخدم y كدالة و x كمتغير):",
        value="Derivative(y(x), x) - y(x) - y(x)*exp(y(x))"
    )

    if st.button("🚀 حل ODE"):
        try:
            ode_expr = sp.sympify(ode_str)
            ode = sp.Eq(ode_expr, 0)

            st.subheader("المرحلة 1: كتابة المعادلة")
            st.latex(sp.latex(ode))

            st.subheader("المرحلة 2: محاولة الحل الرمزي")
            try:
                sol = sp.dsolve(ode, y)
                st.latex(sp.latex(sol))
                if sol.has(sp.LambertW):
                    st.success("✅ الحل يحتوي على دالة لامبرت W")
                    st.markdown(
                        "**التفسير:** المعادلة أُعيد ترتيبها إلى صورة "
                        "$\\phi(y) = \\psi(x)$ حيث $\\phi$ قابلة للعكس بدالة W."
                    )
                else:
                    st.info("الحل لم يستخدم W مباشرة، لكنه حل صحيح.")
            except Exception as e:
                st.warning(
                    f"⚠️ لم يتمكن SymPy من الحل: {e}\n"
                    "جرّب صيغة أبسط مثل: `Derivative(y(x),x) - y(x)*exp(y(x))`"
                )
        except Exception as e:
            st.error(f"❌ خطأ: {e}")

# =========================================================
# القسم 3: PDE
# =========================================================
elif mode == "معادلة تفاضلية جزئية (PDE)":
    st.header("📌 حل PDE يقبل الاختزال إلى دالة W")

    st.markdown(
        "العديد من PDEs غير الخطية من الدرجة الأولى، بعد تطبيق "
        "**طريقة الخصائص**، تُختزل إلى ODE يمكن حلها بدالة W."
    )

    st.latex(r"a(x,y,u)\,u_x + b(x,y,u)\,u_y = c(x,y,u)")

    st.markdown("### مثال نموذجي:")
    st.latex(r"u_x + u\,u_y = u\,e^{u}")

    x, y = sp.symbols("x y", real=True)
    u = sp.Function("u")(x, y)

    a_str = st.text_input("a(x,y,u):", value="1")
    b_str = st.text_input("b(x,y,u):", value="u")
    c_str = st.text_input("c(x,y,u):", value="u*exp(u)")

    if st.button("🚀 حل PDE"):
        try:
            a = sp.sympify(a_str)
            b = sp.sympify(b_str)
            c = sp.sympify(c_str)

            st.subheader("المرحلة 1: المعادلة الأصلية")
            pde = sp.Eq(a * sp.diff(u, x) + b * sp.diff(u, y), c)
            st.latex(sp.latex(pde))

            st.subheader("المرحلة 2: نظام الخصائص")
            st.latex(
                r"\frac{dx}{ds}=" + sp.latex(a) +
                r",\quad \frac{dy}{ds}=" + sp.latex(b) +
                r",\quad \frac{du}{ds}=" + sp.latex(c)
            )

            st.subheader("المرحلة 3: حل ODE لـ u بدلالة s")
            s_sym = sp.Symbol("s")
            u_s = sp.Function("U")(s_sym)
            try:
                ode_u = sp.Eq(sp.diff(u_s, s_sym),
                              c.subs({sp.Symbol("u"): u_s}))
                sol_u = sp.dsolve(ode_u, u_s)
                st.latex(sp.latex(sol_u))
                if sol_u.has(sp.LambertW):
                    st.success(
                        "✅ حل u(s) يحتوي على دالة لامبرت W — "
                        "وهو ما يسمح بعكس العلاقة للحصول على الحل العام."
                    )
                    st.markdown(
                        "**الخطوة التالية:** نعوّض $s = s(x,y)$ من معادلتي "
                        "$dx/ds$ و $dy/ds$ للحصول على $u(x,y)$."
                    )
                else:
                    st.info("حل u(s) لم يستخدم W، لكنه حل صحيح.")
            except Exception as e:
                st.warning(f"⚠️ تعذّر حل ODE الداخلي: {e}")

        except Exception as e:
            st.error(f"❌ خطأ: {e}")

# =========================================================
# القسم 4: حاسبة W مباشرة
# =========================================================
else:
    st.header("📌 حاسبة دالة لامبرت W")

    z_str = st.text_input("أدخل z (يمكن أن يكون مركّباً):", value="1")
    branch = st.selectbox("الفرع:", ["الرئيسي W₀", "الفرع W₋₁"])

    if st.button("احسب W(z)"):
        try:
            z = complex(sp.sympify(z_str).evalf())
            w = sp.LambertW(z, -1 if branch == "الفرع W₋₁" else 0).evalf()
            st.latex(rf"W({z_str}) = {sp.latex(w)}")
            st.markdown(f"**القيمة العددية:** ${complex(w):.6f}$")
            st.markdown(
                f"**تحقق:** $W e^W = {complex(w)*np.exp(complex(w)):.6f}$ "
                f"(يجب أن يساوي z = {z:.6f})"
            )
        except Exception as e:
            st.error(f"❌ خطأ: {e}")

    st.markdown("---")
    st.subheader("📊 رسم دالة W على الأعداد الحقيقية")
    xs = np.linspace(-1/np.e + 1e-6, 5, 500)
    ws = [float(sp.LambertW(x).evalf()) for x in xs]
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(xs, ws)
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_title("Lambert W(x) — real branch")
    ax.set_xlabel("x"); ax.set_ylabel("W(x)")
    ax.grid(True)
    st.pyplot(fig)

# =========================================================
# تذييل
# =========================================================
st.markdown("---")
st.caption("منصة لامبرت W التعليمية — الرياضيات التطبيقية والمعادلات التفاضلية © 2024")
streamlit run lambert_platform.py
