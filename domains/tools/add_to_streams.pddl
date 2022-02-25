
(:stream trust-pick-model
  :inputs (?ot ?pt)
  :domain (and (Block ?ot) (Pose ?ot ?pt))
  :certified (TrustPickModel ?ot ?pt)
)
(:stream trust-contact-model
  :inputs (?o1 ?o2 ?p1 ?p2 ?c)
  :domain (and (Tool ?o1) (Block ?o2) (Pose ?o2 ?p1) (Pose ?o2 ?p2) (Contact ?o1 ?o2 ?c))
  :certified (TrustContactModel ?o1 ?o2 ?p1 ?p2 ?c)
)
