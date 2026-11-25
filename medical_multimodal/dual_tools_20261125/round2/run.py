import os
if os.environ['STUDY']=='fundus':
    from eye import main
else:
    from med import main
main()
