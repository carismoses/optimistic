(define (domain tools-domain)
  (:requirements :strips :equality)
  (:predicates
    ; Types
    (Block ?b)
    (Tool ?to)
    (Table ?ta)

    (Grasp ?o ?g)
    (Conf ?q)
    (Pose ?o ?p)
    (Contact ?o1 ?o2 ?c)

    (FreeMotion ?q1 ?q2 ?t)
    (HoldingMotion ?o ?g ?q1 ?q2 ?t)
    (ContactMotion ?o1 ?g ?o2 ?p1 ?p2 ?c ?q1 ?q2 ?q3 ?t)
    (PickKin ?o ?p ?g ?q1 ?q2 ?t)
    (PlaceKin ?o ?p ?g ?q1 ?q2 ?t)
    (Supported ?ot ?pt ?ob ?pb)

    ; Fluents
    (On ?ot ?ob)
    (AtConf ?q)
    (HandEmpty)
    (AtGrasp ?o ?g)
    (AtPose ?o ?p)
    (FreeObj ?o)
  )

  ; Move in free space while not holing anything
  (:action move_free
    :parameters (?q1 ?q2 ?t)
    :precondition (and (FreeMotion ?q1 ?q2 ?t)
                       (AtConf ?q1)
                       (HandEmpty))
    :effect (and (AtConf ?q2)
                 (not (AtConf ?q1)))
  )

  ; Move while holding Object ?o in Grasp ?g
  (:action move_holding
    :parameters (?o ?g ?q1 ?q2 ?t)
    :precondition (and (HoldingMotion ?o ?g ?q1 ?q2 ?t)
                       (AtConf ?q1)
                       (AtGrasp ?o ?g))
    :effect (and (AtConf ?q2)
                 (not (AtConf ?q1)))
  )

  ; Pick up Object ?ot at Pose ?pt from Object ?ob
  (:action pick
    :parameters (?ot ?pt ?ob ?g ?q1 ?q2 ?t)
    :precondition (and (PickKin ?ot ?pt ?g ?q1 ?q2 ?t)
                       (AtPose ?ot ?pt)
                       (HandEmpty)
                       (AtConf ?q1)
                       (On ?ot ?ob)
                       (FreeObj ?ot))
    :effect (and (AtGrasp ?ot ?g)
                 (AtConf ?q2)
                 (not (AtConf ?q1))
                 (not (AtPose ?ot ?pt))
                 (not (HandEmpty))
                 (not (On ?ot ?ob)))
  )

  ; Place Object ?ot at Pose ?pt on Object ?ob which is at Pose ?pb
  (:action place
    :parameters (?ot ?pt ?ob ?pb ?g ?q1 ?q2 ?t)
    :precondition (and (PlaceKin ?ot ?pt ?g ?q1 ?q2 ?t)
                       (AtGrasp ?ot ?g)
                       (AtConf ?q1)
                       (Supported ?ot ?pt ?ob ?pb)
                       (AtPose ?ob ?pb))
    :effect (and (AtPose ?ot ?pt)
                 (HandEmpty)
                 (AtConf ?q2)
                 (not (AtConf ?q1))
                 (not (AtGrasp ?ot ?g))
                 (On ?ot ?ob))
  )

  ; Make contact ?c between ?o1 being held by robot and ?o2 (at ?p2)
  ; ?q1, ?q2, ?q3 are initial, approach, and final push conf
  ; robot reverses back to approach (?q2) before proceeding with plan
  (:action move_contact
    :parameters (?o1 ?g ?o2 ?p1 ?p2 ?c ?q1 ?q2 ?q3 ?t)
    :precondition (and (ContactMotion ?o1 ?g ?o2 ?p1 ?p2 ?c ?q1 ?q2 ?q3 ?t)
                       (AtConf ?q1)
                       (AtPose ?o2 ?p1)
                       (AtGrasp ?o1 ?g)
                       (FreeObj ?o2))
    :effect (and (AtPose ?o2 ?p2)
                 (AtConf ?q3)
                 (not (AtPose ?o2 ?p1))
                 (not (AtConf ?q1)))
  )
)
