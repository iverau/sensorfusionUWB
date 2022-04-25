
class Brøk:

    def __init__(self, nominator, denominator) -> None:
        self.nominator = nominator
        self.denominator = denominator

    def add_number(self, number: int):
        self.nominator += number*self.denominator

    def add_brok(self, brok):
        if brok.denominator == self.denominator:
            self.nominator += brok.nominator
        else:
            self.nominator = self.nominator*brok.denominator + brok.nominator*self.denominator
            self.denominator = self.denominator * brok.denominator
