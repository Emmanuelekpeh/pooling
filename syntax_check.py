import ast

def check_syntax(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        print(f"No syntax errors found in {filename}")
        return True
    except SyntaxError as e:
        print(f"Syntax error in {filename} at line {e.lineno}, column {e.offset}")
        print(f"Error message: {e.msg}")
        print(f"Line with error: {e.text.strip()}")
        return False
    except Exception as e:
        print(f"Error while parsing {filename}: {str(e)}")
        return False

if __name__ == "__main__":
    check_syntax('train_integrated_fast.py') 