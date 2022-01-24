class DataFrame:

    def __init__(self, topic, msg, t) -> None:
        self.topic = topic
        self.msg = msg
        self.time = t

    def __repr__(self) -> str:
        return f"DataFrame[Topic={self.topic}, time={self.time}]" 