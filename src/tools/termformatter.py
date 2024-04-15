import os    

def title(title: str):
    # Get the size of the terminal
    try:
        columns, _ = os.get_terminal_size()
    except OSError:
        columns = 80

    # Print a big, pretty formatted title
    print("\n" + "=" * columns)
    print(title.center(columns, " "))
    print("=" * columns + "\n")