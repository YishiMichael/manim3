from typing import ClassVar


class Palette:
    __slots__ = ()

    BLUE_E: ClassVar[str] = "#1C758A"
    BLUE_D: ClassVar[str] = "#29ABCA"
    BLUE_C: ClassVar[str] = "#58C4DD"
    BLUE_B: ClassVar[str] = "#9CDCEB"
    BLUE_A: ClassVar[str] = "#C7E9F1"
    TEAL_E: ClassVar[str] = "#49A88F"
    TEAL_D: ClassVar[str] = "#55C1A7"
    TEAL_C: ClassVar[str] = "#5CD0B3"
    TEAL_B: ClassVar[str] = "#76DDC0"
    TEAL_A: ClassVar[str] = "#ACEAD7"
    GREEN_E: ClassVar[str] = "#699C52"
    GREEN_D: ClassVar[str] = "#77B05D"
    GREEN_C: ClassVar[str] = "#83C167"
    GREEN_B: ClassVar[str] = "#A6CF8C"
    GREEN_A: ClassVar[str] = "#C9E2AE"
    YELLOW_E: ClassVar[str] = "#E8C11C"
    YELLOW_D: ClassVar[str] = "#F4D345"
    YELLOW_C: ClassVar[str] = "#FFFF00"
    YELLOW_B: ClassVar[str] = "#FFEA94"
    YELLOW_A: ClassVar[str] = "#FFF1B6"
    GOLD_E: ClassVar[str] = "#C78D46"
    GOLD_D: ClassVar[str] = "#E1A158"
    GOLD_C: ClassVar[str] = "#F0AC5F"
    GOLD_B: ClassVar[str] = "#F9B775"
    GOLD_A: ClassVar[str] = "#F7C797"
    RED_E: ClassVar[str] = "#CF5044"
    RED_D: ClassVar[str] = "#E65A4C"
    RED_C: ClassVar[str] = "#FC6255"
    RED_B: ClassVar[str] = "#FF8080"
    RED_A: ClassVar[str] = "#F7A1A3"
    MAROON_E: ClassVar[str] = "#94424F"
    MAROON_D: ClassVar[str] = "#A24D61"
    MAROON_C: ClassVar[str] = "#C55F73"
    MAROON_B: ClassVar[str] = "#EC92AB"
    MAROON_A: ClassVar[str] = "#ECABC1"
    PURPLE_E: ClassVar[str] = "#644172"
    PURPLE_D: ClassVar[str] = "#715582"
    PURPLE_C: ClassVar[str] = "#9A72AC"
    PURPLE_B: ClassVar[str] = "#B189C6"
    PURPLE_A: ClassVar[str] = "#CAA3E8"
    GREY_E: ClassVar[str] = "#222222"
    GREY_D: ClassVar[str] = "#444444"
    GREY_C: ClassVar[str] = "#888888"
    GREY_B: ClassVar[str] = "#BBBBBB"
    GREY_A: ClassVar[str] = "#DDDDDD"
    WHITE: ClassVar[str] = "#FFFFFF"
    BLACK: ClassVar[str] = "#000000"
    GREY_BROWN: ClassVar[str] = "#736357"
    DARK_BROWN: ClassVar[str] = "#8B4513"
    LIGHT_BROWN: ClassVar[str] = "#CD853F"
    PINK: ClassVar[str] = "#D147BD"
    LIGHT_PINK: ClassVar[str] = "#DC75CD"
    GREEN_SCREEN: ClassVar[str] = "#00FF00"
    ORANGE: ClassVar[str] = "#FF862F"

    BLUE: ClassVar[str] = BLUE_C
    TEAL: ClassVar[str] = TEAL_C
    GREEN: ClassVar[str] = GREEN_C
    YELLOW: ClassVar[str] = YELLOW_C
    GOLD: ClassVar[str] = GOLD_C
    RED: ClassVar[str] = RED_C
    MAROON: ClassVar[str] = MAROON_C
    PURPLE: ClassVar[str] = PURPLE_C
    GREY: ClassVar[str] = GREY_C

    def __new__(cls):
        raise TypeError
