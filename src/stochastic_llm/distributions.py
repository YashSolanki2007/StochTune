"""Target distributions, canonical numeric formats, and sampling prompts."""

from scipy import stats

TRAIN_FAMILIES = ("normal", "uniform", "exponential", "beta", "gamma", "binomial")
HELD_OUT_FAMILIES = ("weibull", "poisson")
DISCRETE_FAMILIES = {"binomial", "poisson"}


def distribution(case):
    p = case["parameters"]
    family = case["family"]
    if family == "normal":
        return stats.norm(loc=p["mean"], scale=p["std"])
    if family == "uniform":
        return stats.uniform(loc=p["low"], scale=p["high"] - p["low"])
    if family == "exponential":
        return stats.expon(scale=1 / p["rate"])
    if family == "beta":
        return stats.beta(p["alpha"], p["beta"])
    if family == "gamma":
        return stats.gamma(p["shape"], scale=p["scale"])
    if family == "binomial":
        return stats.binom(p["n"], p["p"])
    if family == "weibull":
        return stats.weibull_min(p["shape"], scale=p["scale"])
    if family == "poisson":
        return stats.poisson(p["rate"])
    raise ValueError(f"Unknown family: {family}")


def format_number(value, family, decimals=5):
    if family in DISCRETE_FAMILIES:
        return str(int(value))
    value = round(float(value), decimals)
    if value == 0:
        value = 0.0
    return f"{value:.{decimals}f}"


def sample_parameters(family, rng, outside=False):
    def draw(low, high):
        return round(float(rng.uniform(low, high)), 5)

    if family == "normal":
        return {
            "mean": draw(6, 9) if outside else draw(-5, 5),
            "std": draw(3.5, 5) if outside else draw(0.3, 3),
        }
    if family == "uniform":
        low = draw(4, 7) if outside else draw(-5, 2)
        width = draw(9, 12) if outside else draw(0.5, 8)
        return {"low": low, "high": round(low + width, 5)}
    if family == "exponential":
        return {"rate": draw(3.5, 5) if outside else draw(0.2, 3)}
    if family == "beta":
        return {
            "alpha": draw(6, 9) if outside else draw(0.5, 5),
            "beta": draw(6, 9) if outside else draw(0.5, 5),
        }
    if family == "gamma":
        return {
            "shape": draw(6, 9) if outside else draw(0.5, 5),
            "scale": draw(3.5, 5) if outside else draw(0.3, 3),
        }
    if family == "binomial":
        return {
            "n": int(rng.integers(55, 81) if outside else rng.integers(5, 51)),
            "p": draw(0.1, 0.9),
        }
    if family == "weibull":
        return {"shape": draw(0.6, 3), "scale": draw(0.5, 4)}
    if family == "poisson":
        return {"rate": draw(0.5, 20)}
    raise ValueError(f"Unknown family: {family}")


def prompt_for(case, template=0):
    p = case["parameters"]
    descriptions = {
        "normal": lambda: (
            f"a normal distribution with mean {p['mean']} and standard deviation {p['std']}"
        ),
        "uniform": lambda: f"a uniform distribution on [{p['low']}, {p['high']}]",
        "exponential": lambda: f"an exponential distribution with rate {p['rate']}",
        "beta": lambda: f"a beta distribution with alpha {p['alpha']} and beta {p['beta']}",
        "gamma": lambda: f"a gamma distribution with shape {p['shape']} and scale {p['scale']}",
        "binomial": lambda: (
            f"a binomial distribution with n {p['n']} and success probability {p['p']}"
        ),
        "weibull": lambda: f"a Weibull distribution with shape {p['shape']} and scale {p['scale']}",
        "poisson": lambda: f"a Poisson distribution with rate {p['rate']}",
    }
    description = descriptions[case["family"]]()
    if template == 0:
        return f"Sample exactly one number from {description}. Output only the number.\nAnswer:\n"
    if template == 1:
        return (
            f"Generate one random draw from {description}. "
            "Return only its numeric value.\nAnswer:\n"
        )
    if template == 2:
        return (
            f"I need a single independent observation drawn according to {description}.\nNumber:\n"
        )
    raise ValueError(f"Unknown prompt template: {template}")
