
from tools import lookup_ticket

print("TEST 1: Existing ticket")
print(lookup_ticket("TKT-1001"))

print("\nTEST 2: Missing ticket")
print(lookup_ticket("TKT-9999"))
