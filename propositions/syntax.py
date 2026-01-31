# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (c) Yannai A. Gonczarowski and Noam Nisan, 2017-2022
# File name: propositions/syntax.py

"""Syntactic handling of propositional formulas."""

from __future__ import annotations
from functools import lru_cache
from typing import Mapping, Optional, Set, Tuple, Union

from logic_utils import frozen, memoized_parameterless_method

@lru_cache(maxsize=100) # Cache the return value of is_variable
def is_variable(string: str) -> bool:
    """Checks if the given string is a variable name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a variable name, ``False`` otherwise.
    """
    return string[0] >= 'p' and string[0] <= 'z' and \
           (len(string) == 1 or string[1:].isdecimal())

@lru_cache(maxsize=100) # Cache the return value of is_constant
def is_constant(string: str) -> bool:
    """Checks if the given string is a constant.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a constant, ``False`` otherwise.
    """
    return string == 'T' or string == 'F'

@lru_cache(maxsize=100) # Cache the return value of is_unary
def is_unary(string: str) -> bool:
    """Checks if the given string is a unary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a unary operator, ``False`` otherwise.
    """
    return string == '~'

@lru_cache(maxsize=100) # Cache the return value of is_binary
def is_binary(string: str) -> bool:
    """Checks if the given string is a binary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a binary operator, ``False`` otherwise.
    """
    return string == '&' or string == '|' or string == '->'
    # For Chapter 3:
    # return string in {'&', '|',  '->', '+', '<->', '-&', '-|'}

@frozen
class Formula:
    """An immutable propositional formula in tree representation, composed from
    variable names, and operators applied to them.

    Attributes:
        root (`str`): the constant, variable name, or operator at the root of
            the formula tree.
        first (`~typing.Optional`\\[`Formula`]): the first operand of the root,
            if the root is a unary or binary operator.
        second (`~typing.Optional`\\[`Formula`]): the second operand of the
            root, if the root is a binary operator.
    """
    root: str
    first: Optional[Formula]
    second: Optional[Formula]

    def __init__(self, root: str, first: Optional[Formula] = None,
                 second: Optional[Formula] = None):
        """Initializes a `Formula` from its root and root operands.

        Parameters:
            root: the root for the formula tree.
            first: the first operand for the root, if the root is a unary or
                binary operator.
            second: the second operand for the root, if the root is a binary
                operator.
        """
        if is_variable(root) or is_constant(root):
            assert first is None and second is None
            self.root = root
        elif is_unary(root):
            assert first is not None and second is None
            self.root, self.first = root, first
        else:
            assert is_binary(root)
            assert first is not None and second is not None
            self.root, self.first, self.second = root, first, second

    @memoized_parameterless_method
    def __repr__(self) -> str:
        """Computes the string representation of the current formula.

        Returns:
            The standard string representation of the current formula.
        """
        if is_variable(self.root) or is_constant(self.root):
            return self.root
        if is_unary(self.root):
            return self.root + str(getattr(self, 'first'))
        return '(' + str(getattr(self, 'first')) + self.root + str(getattr(self, 'second')) + ')'
        # Task 1.1

    def __eq__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is a `Formula` object that equals the
            current formula, ``False`` otherwise.
        """
        return isinstance(other, Formula) and str(self) == str(other)

    def __ne__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is not a `Formula` object or does not
            equal the current formula, ``False`` otherwise.
        """
        return not self == other

    def __hash__(self) -> int:
        return hash(str(self))

    @memoized_parameterless_method
    def variables(self) -> Set[str]:
        """Finds all variable names in the current formula.

        Returns:
            A set of all variable names used in the current formula.
        """
        res: Set[str] = set()
        def walk(node: 'Formula'):
            if node is None:
                return
            if node.first is None and node.second is None:
                if is_variable(node.root):
                    res.add(node.root)
                return
            walk(node.first)
            walk(node.second)
        walk(self)
        return res
        # Task 1.2

    @memoized_parameterless_method
    def operators(self) -> Set[str]:
        """Finds all operators in the current formula.

        Returns:
            A set of all operators (including ``'T'`` and ``'F'``) used in the
            current formula.
        """
        ops: Set[str] = set()
        def walk(node: 'Formula'):
            if node is None:
                return
            if node.first is None and node.second is None:
                if is_constant(node.root):
                    ops.add(node.root)
            else:
                ops.add(node.root)
            walk(node.first)
            walk(node.second)
        walk(self)
        return ops
        # Task 1.3
        
    @staticmethod
    def _parse_prefix(string: str) -> Tuple[Union[Formula, None], str]:
        """Parses a prefix of the given string into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A pair of the parsed formula and the unparsed suffix of the string.
            If the given string has as a prefix a variable name (e.g.,
            ``'x12'``) or a unary operator followed by a variable name, then the
            parsed prefix will include that entire variable name (and not just a
            part of it, such as ``'x1'``). If no prefix of the given string is a
            valid standard string representation of a formula then returned pair
            should be of ``None`` and an error message, where the error message
            is a string with some human-readable content.
        """
        import re
        var_re = re.compile(r'^[p-z][0-9]*')
        const_re = re.compile(r'^[TF]')
        s = string
        if s == "":
            return None, "empty string"
        m = var_re.match(s)
        if m:
            name = m.group(0)
            return Formula(name), s[m.end():]
        m = const_re.match(s)
        if m:
            c = m.group(0)
            return Formula(c), s[m.end():]
        if s[0] != '(':
            return None, "expected '(' or variable/constant at start"
        idx = 1
        def begins_formula(ch):
            return ch == '(' or ('p' <= ch <= 'z') or ch in ('T','F')
        if idx >= len(s):
            return None, "incomplete parenthesis"
        if begins_formula(s[idx]):
            left, rem = Formula._parse_prefix(s[idx:])
            if left is None:
                return None, "failed parsing left subformula: " + rem
            if rem == "":
                return None, "missing operator and right subformula"
            op_chars = []
            i = 0
            while i < len(rem) and not begins_formula(rem[i]) and rem[i] != ')':
                op_chars.append(rem[i]); i += 1
            if len(op_chars) == 0:
                return None, "missing operator between subformulas"
            op = ''.join(op_chars)
            right, rem2 = Formula._parse_prefix(rem[i:])
            if right is None:
                return None, "failed parsing right subformula: " + rem2
            if rem2 == "" or rem2[0] != ')':
                return None, "missing closing ')' after binary subformula"
            return Formula(op, left, right), rem2[1:]
        else:
            i = idx
            op_chars = []
            while i < len(s) and not begins_formula(s[i]) and s[i] != ')':
                op_chars.append(s[i]); i += 1
            if len(op_chars) == 0:
                return None, "missing unary operator token"
            op = ''.join(op_chars)
            child, rem = Formula._parse_prefix(s[i:])
            if child is None:
                return None, "failed parsing unary child: " + rem
            if rem == "" or rem[0] != ')':
                return None, "missing closing ')' after unary"
            return Formula(op, child), rem[1:]
        # Task 1.4

    @staticmethod
    def is_formula(string: str) -> bool:
        """Checks if the given string is a valid representation of a formula.

        Parameters:
            string: string to check.

        Returns:
            ``True`` if the given string is a valid standard string
            representation of a formula, ``False`` otherwise.
        """
        parsed, rem = Formula._parse_prefix(string)
        return parsed is not None and rem == ""
        # Task 1.5
        
    @staticmethod
    def parse(string: str) -> Formula:
        """Parses the given valid string representation into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A formula whose standard string representation is the given string.
        """
        assert Formula.is_formula(string)
        parsed, rem = Formula._parse_prefix(string)
        return parsed
        # Task 1.6

    def polish(self) -> str:
        """Computes the polish notation representation of the current formula.

        Returns:
            The polish notation representation of the current formula.
        """
        if self.first is None and self.second is None:
            return self.root
        if self.second is None:
            return self.root + ' ' + self.first.polish()
        return self.root + ' ' + self.first.polish() + ' ' + self.second.polish()
        # Optional Task 1.7

    @staticmethod
    def parse_polish(string: str) -> Formula:
        """Parses the given polish notation representation into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A formula whose polish notation representation is the given string.
        """
        import re
        tokens = string.strip().split()
        if tokens == []:
            raise ValueError("empty string")
        def parse_from(i):
            if i >= len(tokens):
                return None, i, "unexpected end of tokens"
            tok = tokens[i]
            if re.match(r'^[p-z][0-9]*$', tok) or tok in ('T','F'):
                return Formula(tok), i+1, None
            child, j, err = parse_from(i+1)
            if child is None:
                return None, i, "failed parsing operand for operator " + tok
            second, k, err2 = parse_from(j)
            if second is None:
                return Formula(tok, child), j, None
            return Formula(tok, child, second), k, None
        formula, next_idx, err = parse_from(0)
        if formula is None or next_idx != len(tokens):
            raise ValueError("invalid polish string: " + (err or "parse incomplete"))
        return formula
        # Optional Task 1.8

    def substitute_variables(self, substitution_map: Mapping[str, Formula]) -> \
            Formula:
        """Substitutes in the current formula, each variable name `v` that is a
        key in `substitution_map` with the formula `substitution_map[v]`.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.

        Returns:
            The formula resulting from performing all substitutions. Only
            variable name occurrences originating in the current formula are
            substituted (i.e., variable name occurrences originating in one of
            the specified substitutions are not subjected to additional
            substitutions).

        Examples:
            >>> Formula.parse('((p->p)|r)').substitute_variables(
            ...     {'p': Formula.parse('(q&r)'), 'r': Formula.parse('p')})
            (((q&r)->(q&r))|p)
        """
        for variable in substitution_map:
            assert is_variable(variable)
        # Task 3.3

    def substitute_operators(self, substitution_map: Mapping[str, Formula]) -> \
            Formula:
        """Substitutes in the current formula, each constant or operator `op`
        that is a key in `substitution_map` with the formula
        `substitution_map[op]` applied to its (zero or one or two) operands,
        where the first operand is used for every occurrence of ``'p'`` in the
        formula and the second for every occurrence of ``'q'``.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.

        Returns:
            The formula resulting from performing all substitutions. Only
            operator occurrences originating in the current formula are
            substituted (i.e., operator occurrences originating in one of the
            specified substitutions are not subjected to additional
            substitutions).

        Examples:
            >>> Formula.parse('((x&y)&~z)').substitute_operators(
            ...     {'&': Formula.parse('~(~p|~q)')})
            ~(~~(~x|~y)|~~z)
        """
        for operator in substitution_map:
            assert is_constant(operator) or is_unary(operator) or \
                   is_binary(operator)
            assert substitution_map[operator].variables().issubset({'p', 'q'})
        # Task 3.4
