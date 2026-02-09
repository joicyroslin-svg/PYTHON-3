import math


def area_of_circle(radius: float) -> float:
    """
    This function calculates the area of a circle given a radius.

    Parameters
    ----------
    radius : float
        The radius of the circle.

    Returns
    -------
    float
        The area of the circle.
    """
    radius = float(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    return math.pi * (radius ** 2)


def _test_area_of_circle() -> None:
    assert area_of_circle(0) == 0.0
    assert round(area_of_circle(1), 12) == round(math.pi, 12)
    assert round(area_of_circle(2), 12) == round(4 * math.pi, 12)
    try:
        area_of_circle(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative radius should raise ValueError")


if __name__ == "__main__":
    print("Area with radius 3:", area_of_circle(3))
    _test_area_of_circle()
    print("Tests passed.")
