"""Apply an eligible, frozen instruction at the ordinary HCL localization entry."""
from runtime import EpisodeTools


class GuidedEpisodeTools(EpisodeTools):
    def __init__(self, *args, guidance, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(guidance, str) or not 1 <= len(guidance.strip()) <= 600:
            raise ValueError("localization guidance must be nonblank and fit the tool input")
        self.guidance = guidance.strip()
        self.guided_calls = 0

    def execute(self, name, args, source="controller"):
        if name == "localize" and source == "controller" and isinstance(args, dict):
            original = args.get("instruction", "")
            if isinstance(original, str):
                text = (original.strip() + "\n" + self.guidance).strip() if original.strip() else self.guidance
                if len(text) <= 600:
                    args = {**args, "instruction": text}
                    self.guided_calls += 1
        return super().execute(name, args, source)
