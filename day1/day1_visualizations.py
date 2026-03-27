import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def my_first_plot():
#A 2D Graphics Environment
    fig,ax = plt.subplots(2,2)
    ax[0,0].plot(2.5,5,"r-o")
    ax[0,0].set_title('My first plot')
    plt.show()

    plt.plot(2.5, 5, "r-o")
    plt.title('My first plot')
    plt.show()


def mandelbrot(h, w, maxit=20, r=2):
    """Returns an image of the Mandelbrot fractal of size (h,w)."""
    x = np.linspace(-2.5, 1.5, 4*h+1)
    y = np.linspace(-1.5, 1.5, 3*w+1)
    A, B = np.meshgrid(x, y)
    C = A + B*1j
    z = np.zeros_like(C)
    divtime = maxit + np.zeros(z.shape, dtype=int)

    for i in range(maxit):
        z = z**2 + C
        diverge = abs(z) > r                    # who is diverging
        div_now = diverge & (divtime == maxit)  # who is diverging now
        divtime[div_now] = i                    # note when
        z[diverge] = r                          # avoid diverging too much

    return divtime
# plt.clf()
# plt.imshow(mandelbrot(400, 400))
# plt.show()

tips =sns.load_dataset("tips")
sns.regplot(x="total_bill", y="tip", data=tips)
plt.title('Total Bill vs Tip with Regression Line')
# plt.show()

# Set Seaborn style for better default aesthetics
sns.set_style("ticks")

# Create a Matplotlib figure and axes
fig, ax = plt.subplots()

# Use Seaborn to plot a boxplot on the specified axes
sns.boxplot(x='day', y='total_bill', data=tips, ax=ax,linecolor='skyblue', linewidth=2)

# Use Matplotlib to add custom details
ax.set_title('Total Bill Distribution by Day', fontsize=16)
ax.set_xlabel('Day of the Week', fontsize=12)
ax.set_ylabel('Total Bill ($)', fontsize=12)

plt.show()