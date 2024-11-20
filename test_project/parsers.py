import re

from declaration import FunctionDeclaration, TypeDeclaration, ObjectDeclaration, ClassDeclaration
from declaration import Parameter

modifiers = [
    "abstract",
    "actual",
    "annotation",
    "companion",
    "const",
    "crossinline",
    "data",
    "enum",
    "expect",
    "external",
    "final",
    "infix",
    "inline",
    "inner",
    "internal",
    "lateinit",
    "noinline",
    "open",
    "operator",
    "out",
    "override",
    "private",
    "protected",
    "public",
    "reified",
    "sealed",
    "suspend",
    "tailrec",
    "vararg"
]

def find_declarations(tokens):
    declarations = (
            find_all_function_declarations(tokens) +
            find_all_type_alises(tokens) +
            find_all_objects(tokens) +
            find_all_class_declarations(tokens)
    )
    return declarations


def find_declarations_by_type(tokens, keyword, declaration_type):
    declarations = []
    for i in range(len(tokens)):
        if tokens[i] == keyword:
            name = tokens[i + 1]
            j = i - 1
            current_modifiers = []
            while j >= 0 and tokens[j] in modifiers:
                current_modifiers.insert(0, tokens[j])
                j -= 1
            declaration = declaration_type(name=name, modifiers=current_modifiers)
            declarations.append(declaration)
    return declarations



def find_all_objects(tokens):
    return find_declarations_by_type(tokens, 'object', ObjectDeclaration)


def find_all_type_alises(tokens):
    return find_declarations_by_type(tokens, 'typealias', TypeDeclaration)


def find_all_class_declarations(tokens):
    return find_declarations_by_type(tokens, 'class', ClassDeclaration)


def find_all_function_declarations(tokens):
    declarations_list = []
    for fun in parse_functions_from_tokens(tokens):
        parsed_function = parse_function_from_string(fun)
        name = parsed_function[0][1].split('(')[0]
        signature = parsed_function[0][1]
        parameters = extract_parameters_from_signature(signature)
        match_type = re.compile(r'\):\s*(\w+)\s*$').search(signature)
        if match_type:
            returntype = match_type.group(1)
        else:
            returntype = 'Unit'
        declaration = FunctionDeclaration(name=name, parameters=parameters, returnType=returntype, body=fun)
        insides = parsed_function[0][2].strip()[1:].strip() + '}'
        if insides:
            insides_declarations = find_declarations(insides.split())
            if insides_declarations:
                for dec in insides_declarations:
                    declaration.add(dec)
        declarations_list.append(declaration)
    return declarations_list


def extract_parameters_from_signature(signature):
    signature = signature.replace('\n', ' ')
    param_string = re.search(r'\((.*)\)', signature, re.DOTALL)
    if param_string:
        param_string = param_string.group(1)
    else:
        return []
    params = re.findall(r'([\w]+)\s*:\s*([\w]+)(?:\s*=\s*([^,]+))?', param_string)
    parameters = [Parameter(name, type_, default.strip() if default is not None else None) for name, type_, default in
                  params]
    return parameters


def parse_function_from_string(file_contents):
    pattern = r'fun\s+([^\{]+)(\{.*?\})'
    matches = re.findall(pattern, file_contents, re.DOTALL)
    tokens = []
    for match in matches:
        function_signature, function_body = match
        tokens.append(('fun', function_signature.strip(), function_body.strip()))
    return tokens


def parse_functions_from_tokens(tokens):
    sections = []
    inside_fun = False
    brace_balance = 0
    current_section = []
    for token in tokens:
        if token == "fun" and (not inside_fun or brace_balance == 0):
            if inside_fun:
                sections.append(" ".join(current_section))
                current_section = []
            inside_fun = True
            brace_balance = 0
            current_section.append(token)
        elif inside_fun:
            current_section.append(token)
            if token == "{":
                brace_balance += 1
            elif token == "}":
                brace_balance -= 1
                if brace_balance == 0:
                    if not (len(tokens) > tokens.index(token) + 1 and tokens[tokens.index(token) + 1] == "fun"):
                        sections.append(" ".join(current_section))
                        inside_fun = False
                        current_section = []
    if current_section:
        sections.append(" ".join(current_section))
    return sections
