(define (stream ordered-blocks-optimistic-panda)
  ; Generate a motion for the robot in free space while not holding anything
  (:stream plan-free-motion
    :inputs (?q1 ?q2)
    :domain (and (Conf ?q1) (Conf ?q2))
    :fluents (AtPose)
    :outputs (?t)
    :certified (FreeMotion ?q1 ?t ?q2)
  )
  ; Generate a motion for the robot while it is holding Block ?b in Grasp ?g
  (:stream plan-holding-motion
    :inputs (?q1 ?q2 ?b ?g)
    :domain (and (Conf ?q1) (Conf ?q2) (Block ?b) (PickGraspKin ?b ?g ?q1) (PlaceGraspKin ?b ?g ?q2))
    :fluents (AtPose)
    :outputs (?t)
    :certified (HoldingMotion ?q1 ?t ?q2 ?b ?g)
  )
  ; Generate a pick trajectory
  (:stream pick-inverse-kinematics
    :inputs (?b ?p ?g)
    :domain (and (Block ?b) (Pose ?b ?p) (Grasp ?b ?g))
    :outputs (?q1 ?q2 ?t)
    :certified (and (Conf ?q1) (Conf ?q2) (PickGraspKin ?b ?g ?q2) (PickKin ?b ?p ?g ?q1 ?q2 ?t))
  )
  ; Generate a place trajectory
  (:stream place-inverse-kinematics
    :inputs (?b ?p ?g)
    :domain (and (Block ?b) (Pose ?b ?p) (Grasp ?b ?g))
    :outputs (?q1 ?q2 ?t)
    :certified (and (Conf ?q1) (PlaceGraspKin ?b ?g ?q1) (Conf ?q2) (PlaceKin ?b ?p ?g ?q1 ?q2 ?t))
  )
  ; Sample a pose for Block ?bt when placed on top of Block ?bb at Pose ?pb.
  (:stream sample-pose-block
    :inputs (?bt ?bb ?pb)
    :domain (and (Block ?bt) (Block ?bb) (Pose ?bb ?pb))
    :outputs (?pt)
    :certified (and (Pose ?bt ?pt) (Supported ?bt ?pt ?bb ?pb))
  )
  ; Sample a grasp to pick up Block ?b
  (:stream sample-grasp
    :inputs (?b)
    :domain (Block ?b)
    :outputs (?g)
    :certified (Grasp ?b ?g)
  )
)
