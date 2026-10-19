
import matplotlib.pyplot as plt
import numpy as np

from methods import *


def draw_table(data, columns, title, filename, formats=None):
    """Создание таблицы и сохранение её в PNG."""
    fig, ax = plt.subplots(figsize=(12, 2 + 0.45 * len(data)))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=15)

    table = ax.table(
        cellText=data,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    x_0, y_0 = 0.0, 0.0
    x_end = 2.0
    steps = [0.1, 0.05, 0.025, 0.0125]

    methods = {
        "Эйлер": euler_method,
        "Рунге-Кутта 2": rk2_method,
        "Рунге-Кутта 4": rk4_method,
    }

    results = []
    errors_step = {}

    # Расчёты
    for h in steps:
        n = round((x_end - x_0) / h)
        errors_step[h] = {}

        for name, method in methods.items():
            x, y = method(x_0, y_0, h, n)
            y_exact = exact_solution(x)
            error = np.abs(y_exact - y)
            max_error = np.max(error)

            errors_step[h][name] = max_error

            results.append({
                "h": h,
                "n": n,
                "Метод": name,
                "Макс. погрешность": max_error,
            })

    # --------------------------------------------------
    # 1. График решений при h = 0.1
    # --------------------------------------------------
    h = steps[0]
    n = round((x_end - x_0) / h)

    x_fine = np.linspace(x_0, x_end, 500)

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_fine,
        exact_solution(x_fine),
        label="Точное решение",
        linewidth=2.5,
    )

    for name, method in methods.items():
        x, y = method(x_0, y_0, h, n)
        plt.plot(x, y, "o--", markersize=3, label=name)

    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title(f"Сравнение решений при h = {h}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("solutions.png", dpi=300)
    plt.show()

    # --------------------------------------------------
    # 2. График абсолютной погрешности
    # --------------------------------------------------
    plt.figure(figsize=(10, 6))

    for name, method in methods.items():
        x, y = method(x_0, y_0, h, n)
        error = np.abs(exact_solution(x) - y)

        plt.plot(x, error, "o-", markersize=3, label=name)

    plt.xlabel("x")
    plt.ylabel("Абсолютная погрешность")
    plt.title(f"Погрешность методов при h = {h}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("errors.png", dpi=300)
    plt.show()

    # --------------------------------------------------
    # 3. Максимальная погрешность в зависимости от h
    # --------------------------------------------------
    plt.figure(figsize=(10, 6))

    for name in methods:
        errors = [errors_step[h][name] for h in steps]
        plt.loglog(steps, errors, "o-", label=name)

    plt.xlabel("Шаг интегрирования h")
    plt.ylabel("Максимальная абсолютная погрешность")
    plt.title("Зависимость погрешности от шага")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence.png", dpi=300)
    plt.show()

    # --------------------------------------------------
    # 4. Таблица максимальных погрешностей
    # --------------------------------------------------
    error_table = []

    for h in steps:
        row = [f"{h:.4f}", str(round((x_end - x_0) / h))]

        for name in methods:
            row.append(f"{errors_step[h][name]:.6e}")

        error_table.append(row)

    draw_table(
        error_table,
        ["h", "n"] + list(methods.keys()),
        "Максимальные абсолютные погрешности",
        "error_table.png",
    )

    # --------------------------------------------------
    # 5. Таблица фактического порядка сходимости
    # --------------------------------------------------
    order_table = []

    for i in range(len(steps) - 1):
        h = steps[i]
        h_next = steps[i + 1]

        row = [f"{h:.4f} → {h_next:.4f}"]

        for name in methods:
            e1 = errors_step[h][name]
            e2 = errors_step[h_next][name]

            if e1 > 0 and e2 > 0:
                p = np.log(e1 / e2) / np.log(h / h_next)
                row.append(f"{p:.4f}")
            else:
                row.append("—")

        order_table.append(row)

    draw_table(
        order_table,
        ["Изменение шага"] + list(methods.keys()),
        "Фактический порядок сходимости",
        "order_table.png",
    )
