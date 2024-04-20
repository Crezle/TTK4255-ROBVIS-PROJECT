import os    

def title(title: str):
    """Print a big, pretty formatted title.
    
    Args:
        title (str): The title to print.
    """
    try:
        columns, _ = os.get_terminal_size()
    except OSError:
        columns = 80

    # Print a big, pretty formatted title
    print("\n" + "=" * columns)
    print(title.center(columns, " "))
    print("=" * columns + "\n")