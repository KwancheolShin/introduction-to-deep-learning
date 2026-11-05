import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Function
# f(x) = 1/4 x^4 + 1/3 x^3 + 1/3 x^2
# ============================================================

f   = lambda x: 1/4*x**4 + 1/3*x**3 + 1/3*x**2
df  = lambda x: x**3 + x**2 + 2/3*x
ddf = lambda x: 3*x**2 + 2*x + 2/3


# ============================================================
# Initial point
# ============================================================

xk = 0.5

# x-axis for plotting
x = np.linspace(-0.5, 1.3, 500)


# ============================================================
# Figure: 2 x 2 subplots
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# First three axes: Newton steps
step_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]

# Fourth axis: combined plot
ax_all = axes[1, 1]


# Store information for the combined plot
history = []


# ============================================================
# Newton Method: 3 iterations
# ============================================================

for k in range(3):

    # --------------------------------------------------------
    # Newton step
    # --------------------------------------------------------

    x_next = xk - df(xk) / ddf(xk)


    # --------------------------------------------------------
    # Quadratic approximation at x_k
    # --------------------------------------------------------

    q = (
        f(xk)
        + df(xk) * (x - xk)
        + 0.5 * ddf(xk) * (x - xk)**2
    )


    # Minimum value of quadratic approximation
    q_min = (
        f(xk)
        + df(xk) * (x_next - xk)
        + 0.5 * ddf(xk) * (x_next - xk)**2
    )


    # Save for combined plot
    history.append({
        'k': k,
        'xk': xk,
        'x_next': x_next,
        'q': q.copy(),
        'q_min': q_min
    })


    # ========================================================
    # Plot Step k+1
    # ========================================================

    ax = step_axes[k]


    # Original function
    ax.plot(
        x, f(x),
        'k',
        linewidth=2,
        label=r'$f(x)$'
    )


    # Quadratic approximation
    ax.plot(
        x, q,
        'b--',
        linewidth=2,
        label=rf'$q_{k}(x)$'
    )


    # Current point x_k
    ax.plot(
        xk, f(xk),
        'ro',
        markersize=8,
        label=rf'$x_{k}$'
    )


    # Minimum of q_k
    ax.plot(
        x_next, q_min,
        'bs',
        markersize=8,
        label=rf'$x_{k+1}$: minimum of $q_{k}$'
    )


    # Vertical line at x_{k+1}
    ax.axvline(
        x_next,
        color='gray',
        linestyle=':',
        linewidth=1.5
    )


    # Newton step arrow on x-axis
    ax.annotate(
        '',
        xy=(x_next, 0),
        xytext=(xk, 0),
        arrowprops=dict(
            arrowstyle='->',
            linewidth=2
        )
    )


    # Labels
    ax.text(
        xk,
        -0.08,
        rf'$x_{k}={xk:.3f}$',
        ha='center',
        fontsize=10
    )

    ax.text(
        x_next,
        -0.18,
        rf'$x_{k+1}={x_next:.3f}$',
        ha='center',
        fontsize=10
    )


    # --------------------------------------------------------
    # Plot settings
    # --------------------------------------------------------

    ax.axhline(0, color='gray', linewidth=0.8)

    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')

    ax.set_title(
        rf"Newton's Method: Step {k+1}",
        fontsize=14
    )

    ax.set_xlim(-0.5, 1.3)
    ax.set_ylim(-0.3, 2.0)

    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)


    # --------------------------------------------------------
    # Print numerical information
    # --------------------------------------------------------

    print(f"Step {k+1}")
    print(f"x_{k}      = {xk:.6f}")
    print(f"f'(x_{k})  = {df(xk):.6f}")
    print(f"f''(x_{k}) = {ddf(xk):.6f}")
    print(f"x_{k+1}    = {x_next:.6f}")
    print()


    # Update
    xk = x_next


# ============================================================
# Fourth subplot: all three steps together
# ============================================================

# Original function
ax_all.plot(
    x, f(x),
    'k',
    linewidth=2.5,
    label=r'$f(x)$'
)


# Plot all quadratic approximations
linestyles = ['--', '-.', ':']

for i, item in enumerate(history):

    ax_all.plot(
        x,
        item['q'],
        linestyle=linestyles[i],
        linewidth=2,
        label=rf'$q_{i}(x)$'
    )


# Newton iterates on the original function
x_points = [history[0]['xk']]
x_points += [item['x_next'] for item in history]

x_points = np.array(x_points)

ax_all.plot(
    x_points,
    f(x_points),
    'ro-',
    linewidth=1.5,
    markersize=7,
    label='Newton iterates'
)


# Labels x_0, x_1, x_2, x_3
for i, xi in enumerate(x_points):

    ax_all.annotate(
        rf'$x_{i}$',
        xy=(xi, f(xi)),
        xytext=(5, 8),
        textcoords='offset points',
        fontsize=11
    )


# ------------------------------------------------------------
# Combined plot settings
# ------------------------------------------------------------

ax_all.axhline(0, color='gray', linewidth=0.8)

ax_all.set_xlabel('x')
ax_all.set_ylabel('f(x)')

ax_all.set_title(
    "Newton's Method: Steps 1–3",
    fontsize=14
)

ax_all.set_xlim(-0.5, 1.3)
ax_all.set_ylim(-0.3, 2.0)

ax_all.grid(alpha=0.25)
ax_all.legend(fontsize=9)


# ============================================================
# Final layout
# ============================================================

plt.tight_layout()
plt.show()