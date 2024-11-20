import json

from declaration import Declaration, FunctionDeclaration, ClassDeclaration, ObjectDeclaration, PropertyDeclaration, \
    TypeDeclaration
from parsers import find_declarations


class Program:
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            file_content = file.read()

        self.tokens = file_content.split()
        self.file_paths = file_path

        self.declarations = find_declarations(self.tokens)

    def add_declaration(self, declaration: Declaration):
        self.declarations.append(declaration)

    def to_dict(self):
        def declaration_to_dict(declaration):
            if isinstance(declaration, FunctionDeclaration):
                result = {
                    'type': declaration.type.value,
                    'name': declaration.name,
                    'parameters': [
                        {'name': param.name, 'type': param.type, **({'value': param.value} if param.value else {})}
                        for param in declaration.parameters],
                    'returnType': declaration.returnType,
                    'body': declaration.body,
                }
                if declaration.modifiers:
                    result['modifiers'] = [mod for mod in declaration.modifiers]
                if declaration.declarations:
                    result['declarations'] = [declaration_to_dict(decl) for decl in declaration.declarations]
                return result
            elif isinstance(declaration, ClassDeclaration):
                result = {
                    'type': declaration.type.value,
                    'name': declaration.name
                }
                if declaration.declarations:
                    result['declarations'] = [declaration_to_dict(decl) for decl in declaration.declarations]
                if declaration.modifiers:
                    result['modifiers'] = [mod for mod in declaration.modifiers]
                return result
            elif isinstance(declaration, ObjectDeclaration):
                result = {
                    'type': declaration.type.value,
                    'name': declaration.name
                }
                if declaration.declarations:
                    result['declarations'] = [declaration_to_dict(decl) for decl in declaration.declarations]
                if declaration.modifiers:
                    result['modifiers'] = [mod for mod in declaration.modifiers]
                return result
            elif isinstance(declaration, PropertyDeclaration):
                result = {
                    'type': declaration.type.value,
                    'name': declaration.name
                }
                if declaration.modifiers:
                    result['modifiers'] = [mod for mod in declaration.modifiers]
                if declaration.declarations:
                    result['declarations'] = [declaration_to_dict(decl) for decl in declaration.declarations]
                return result
            elif isinstance(declaration, TypeDeclaration):
                result = {
                    'type': declaration.type.value,
                    'name': declaration.name
                }
                if declaration.modifiers:
                    result['modifiers'] = [mod for mod in declaration.modifiers]
                if declaration.declarations:
                    result['declarations'] = [declaration_to_dict(decl) for decl in declaration.declarations]
                return result

        return {"declarations": [declaration_to_dict(decl) for decl in self.declarations]}

    def to_json(self):
        return json.dumps(self.to_dict())

    def export_declarations_to_json(self, file_path):
        json_obj = self.to_json()
        with open(file_path, "w") as outfile:
            outfile.write(json_obj)
        return json_obj
