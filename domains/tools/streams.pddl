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
  ; Sample a contact (relative pose) between ?o1 and ?o2
  (:stream sample-contact
    :inputs (?o1 ?o2)
    :domain (and (Tool ?o1) (Block ?o2))
    :outputs (?c)
    :certified (Contact ?o1 ?o2 ?c)
  )
  ; Generate a motion for the robot to push object ?o1 from ?p1 to ?p2 while holding
  ; object ?o2 in grasp ?g with ?o2 in contact ?c with ?o1
  (:stream plan-contact-motion
    :inputs (?o1 ?c ?p1 ?p2 ?o2 ?g ?q1 ?q2)
    :domain (and (Conf ?q1) (Conf ?q2) (Contact ?o1 ?o2 ?c) (Pose ?o2 ?p1) (Pose ?o2 ?p2) (Grasp ?o2 ?g))
    :fluents (AtPose)
    :outputs (?t)
    :certified (and (ContactMotion ?o1 ?c ?p1 ?p2 ?o2 ?g ?q1 ?q2 ?t))
  )
  ; Generate a motion for the robot to make contact ?c1 between ?o1 at ?p1 and ?o2
  ; which is being held at grasp ?g
  (:stream plan-make-contact-motion
    :inputs (?o1 ?g ?o2 ?p2 ?c)
    :domain (and (Grasp ?o1 ?g) (Block ?o2) (Pose ?o2 ?p2) (Contact ?o1 ?o2 ?c)); TODO: remove Block constraint
    :outputs (?q1 ?q2 ?t)
    :certified (and (Conf ?q1) (Conf ?q2) (MakeContactMotion ?o1 ?g ?o2 ?p2 ?c ?q1 ?q2 ?t))
  )
)
