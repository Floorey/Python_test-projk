from dataclasses import dataclass, field


@dataclass
class Name:
    first: str
    last: str
    full: str = field(init=False)


    def __post_init__(self):
        self.full = f'{self.first} {self.last}'


fullname = Name(first='Lukas', last='Enderle')              

print(fullname.full)


