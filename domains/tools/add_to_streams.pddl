(:stream trust-model
  :inputs (?o1 ?o2 ?p1 ?p2 ?c)
  :domain (and (Tool ?o1) (Block ?o2) (Pose ?o2 ?p1) (Pose ?o2 ?p2) (Contact ?o1 ?o2 ?c))
  :fluents (AtPose AtGrasp AtConf)
  :certified (TrustModel ?o1 ?o2 ?p1 ?p2 ?c)
)
