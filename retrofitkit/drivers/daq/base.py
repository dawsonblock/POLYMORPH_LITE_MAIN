class DAQBase:
    def set_voltage(self, volts: float) -> None:
        raise NotImplementedError

    def read_ai(self) -> float:
        raise NotImplementedError

    def read_di(self, line: int) -> bool:
        raise NotImplementedError
