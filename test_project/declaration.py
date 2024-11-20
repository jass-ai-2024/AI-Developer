from enum import Enum
from typing import List


class DeclarationType(Enum):
    CLASS = "class"
    OBJECT = "object"
    FUNCTION = "function"
    PROPERTY = "property"
    TYPE = "type"


class Parameter:
    def __init__(self, name: str, type: str, value: str):
        self.name = name
        self.type = type
        self.value = value

    def __repr__(self):
        return f"Parameter(name='{self.name}', type='{self.type}', value='{self.value}')"


class Declaration:
    def __init__(self, name: str, type: DeclarationType, declarations, modifiers):
        self.name = name
        self.type = type
        if declarations is None:
            self.declarations = []
        else:
            self.declarations = declarations
        if modifiers is None:
            self.modifiers = []
        else:
            self.modifiers = modifiers

    def add(self, declaration):
        self.declarations.append(declaration)

    def remove(self, declaration):
        self.declarations.remove(declaration)


class ObjectDeclaration(Declaration):
    def __init__(self, name: str, modifiers=None):
        super().__init__(name, declarations=None, type=DeclarationType.OBJECT, modifiers=modifiers)

    def __repr__(self):
        declarations_repr = ', '.join(repr(declaration) for declaration in self.declarations)
        return f"ClassDeclaration(type='{self.type}', name='{self.name}', declarations=[{declarations_repr}])"


class ClassDeclaration(Declaration):
    def __init__(self, name: str, modifiers=None):
        super().__init__(name, declarations=None, type=DeclarationType.CLASS, modifiers=modifiers)

    def __repr__(self):
        declarations_repr = ', '.join(repr(declaration) for declaration in self.declarations)
        return f"ClassDeclaration(type='{self.type}', name='{self.name}', declarations=[{declarations_repr}])"


class PropertyDeclaration(Declaration):
    def __init__(self, name: str, modifiers=None):
        super().__init__(name, declarations=None, type=DeclarationType.PROPERTY, modifiers=modifiers)

    def __repr__(self):
        declarations_repr = ', '.join(repr(declaration) for declaration in self.declarations)
        return f"ClassDeclaration(type='{self.type}', name='{self.name}', declarations=[{declarations_repr}])"


class TypeDeclaration(Declaration):
    def __init__(self, name: str, modifiers=None):
        super().__init__(name, declarations=None, type=DeclarationType.TYPE, modifiers=modifiers)

    def __repr__(self):
        declarations_repr = ', '.join(repr(declaration) for declaration in self.declarations)
        return f"ClassDeclaration(type='{self.type}', name='{self.name}', declarations=[{declarations_repr}])"


class FunctionDeclaration(Declaration):
    def __init__(self, name: str, parameters: List[Parameter], returnType: str, body: str, modifiers=None):
        super().__init__(name, declarations=None, type=DeclarationType.FUNCTION, modifiers=modifiers)
        self.parameters = parameters
        self.returnType = returnType
        self.body = body

    def __repr__(self):
        modifiers_repr = ', '.join(repr(mod) for mod in self.modifiers)
        parameters_repr = ', '.join(repr(param) for param in self.parameters)
        declarations_repr = ', '.join(repr(declaration) for declaration in self.declarations)
        return (f"FunctionDeclaration(type='{self.type}', name='{self.name}', "
                f"modifiers=[{modifiers_repr}], "
                f"parameters=[{parameters_repr}], "
                f"returnType='{self.returnType}', body='{self.body}', declarations=[{declarations_repr}])")
