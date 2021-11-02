(define (stream tools-domain)
  ; Generate a motion for the robot in free space while not holding anything
  (:stream plan-free-motion
    :inputs (?q1 ?q2)
    :domain (and (Conf ?q1) (Conf ?q2))
    :fluents (AtPose)
    :outputs (?t)
    :certified (FreeMotion ?q1 ?t ?q2)
  )
  ; Generate a motion for the robot while it is holding Object ?o in Grasp ?g
  (:stream plan-holding-motion
    :inputs (?q1 ?q2 ?o ?g)
    :domain (and (Conf ?q1) (Conf ?q2) (PickGraspKin ?o ?g ?q1) (PlaceGraspKin ?o ?g ?q2))
    :fluents (AtPose)
    :outputs (?t)
    :certified (HoldingMotion ?q1 ?t ?q2 ?o ?g)
  )
  ; Generate a pick trajectory
  (:stream pick-inverse-kinematics
    :inputs (?o ?p ?g)
    :domain (and (Pose ?o ?p) (Grasp ?o ?g))
    :outputs (?q1 ?q2 ?t)
    :certified (and (Conf ?q1) (Conf ?q2) (PickGraspKin ?o ?g ?q2) (PickKin ?o ?p ?g ?q1 ?q2 ?t))
  )
  ; Generate a place trajectory
  (:stream place-inverse-kinematics
    :inputs (?o ?p ?g)
    :domain (and (Pose ?o ?p) (Grasp ?o ?g))
    :outputs (?q1 ?q2 ?t)
    :certified (and (Conf ?q1) (PlaceGraspKin ?o ?g ?q1) (Conf ?q2) (PlaceKin ?o ?p ?g ?q1 ?q2 ?t))
  )
  ; Sample a pose for Block ?bt when placed on top of Object ?ob at Pose ?pb.
  (:stream sample-pose-block
    :inputs (?bt ?ob ?pb)
    :domain (and (Block ?bt) (Pose ?ob ?pb))
    :outputs (?pt)
    :certified (and (Pose ?bt ?pt) (Supported ?bt ?pt ?ob ?pb))
  )
  ; Sample a grasp to pick up Block ?b
  (:stream sample-block-grasp
    :inputs (?b)
    :domain (Block ?b)
    :outputs (?g)
    :certified (Grasp ?b ?g)
  )
  ; Sample a grasp to pick up Tool ?to
  (:stream sample-tool-grasp
    :inputs (?to)
    :domain (Tool ?to)
    :outputs (?g)
    :certified (Grasp ?to ?g)
  )
)
