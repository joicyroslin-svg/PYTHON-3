def get_number(prompt: str) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid number.")


def main() -> None:
    print("Simple Calculator")
    print("Operations: +  -  *  /")

    a = get_number("Enter the first number: ")
    op = input("Enter operation (+, -, *, /): ").strip()
    b = get_number("Enter the second number: ")

    if op == "+":
        result = a + b
    elif op == "-":
        result = a - b
    elif op == "*":
        result = a * b
    elif op == "/":
        if b == 0:
            print("Cannot divide by zero.")
            return
        result = a / b
    else:
        print("Unknown operation.")
        return

    print(f"Result: {result}")


if __name__ == "__main__":
    main()
