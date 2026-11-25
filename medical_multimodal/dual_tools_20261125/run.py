import os
if os.environ["STUDY"] == "retina":
    from retina import main
else:
    from clinical import main
main()
