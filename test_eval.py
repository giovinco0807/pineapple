import sys; sys.path.insert(0, '.')
from backend.main import evaluate_hand, hand_category

tests = [
    (evaluate_hand(['Kh','Kd','9h','9d','4h'],5) > evaluate_hand(['Kh','Kd','9h','9d','2h'],5), 'KK994>KK992'),
    (evaluate_hand(['Ah','Ad','9h','9d','3h'],5) > evaluate_hand(['Kh','Kd','9h','9d','3h'],5), 'AA99>KK99'),
    (evaluate_hand(['Ah','Kh','9s'],3) > evaluate_hand(['Ah','Qh','9s'],3), 'AK9>AQ9(3card)'),
    (evaluate_hand(['Ah','Ad','9s','8s','7s'],5) > evaluate_hand(['Ah','Ad','9s'],3), 'AA987(5)>AA9(3)'),
    (evaluate_hand(['Ah','Th','8h','6h','3h'],5) > evaluate_hand(['Kh','Th','8h','6h','3h'],5), 'FlushA>FlushK'),
    (evaluate_hand(['Ah','Kd','Qh','Js','Td'],5) > evaluate_hand(['Th','9d','8h','7s','6d'],5), 'StrA>StrT'),
    (hand_category(evaluate_hand(['Kh','Kd','9h','9d','4h'],5)) == 2, 'KK99 cat=TwoPair'),
    (hand_category(evaluate_hand(['Ah','Kh','Qh','Jh','Th'],5)) == 8, 'AKQJT cat=StrFlush'),
    (hand_category(evaluate_hand(['Ah','Ad','9s'],3)) == 1, 'AA9(3) cat=Pair'),
]
passed = 0
for ok, name in tests:
    status = "PASS" if ok else "FAIL"
    print(status + ": " + name)
    if ok:
        passed += 1
print(str(passed) + "/" + str(len(tests)) + " passed")
