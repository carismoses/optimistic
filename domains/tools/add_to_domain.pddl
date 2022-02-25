; Add  (TrustContactModel ?o1 ?o2 ?p1 ?p2 ?c) as a precondition to the move_contact action
; Add (TrustPickModel ?ot ?pt) as a precondition to the pick action

action: pick
pre: (TrustPickModel ?ot ?pt)

action: move_contact
pre: (TrustContactModel ?o1 ?o2 ?p1 ?p2 ?c)
