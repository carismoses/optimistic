(define (domain tools-domain)
  (:requirements :strips :equality)
  (:predicates
    ; Types
    (Block ?b)
    (Tool ?to)
    (Table ?ta)
    (Tunnel ?tu)
    (Patch ?pa)

    (Grasp ?o ?g)
    (Conf ?q)
    (Pose ?o ?p)
    (Contact ?o1 ?o2 ?c)

    (FreeMotion ?q1 ?t ?q2)
    (HoldingMotion ?q1 ?t ?q2 ?o ?g)
    (ContactMotion ?o1 ?c ?p1 ?p2 ?o2 ?g ?q1 ?q2 ?t)
    (MakeContactMotion ?o1 ?g ?o2 ?p ?c ?q1 ?q2 ?t)
    (PickKin ?o ?p ?g ?q1 ?q2 ?t)
    (PlaceKin ?o ?p ?g ?q1 ?q2 ?t)
    (Supported ?ot ?pt ?ob ?pb)

    ; Fluents
    (On ?ot ?ob)
    (Clear ?o)
    (AtConf ?q)
    (HandEmpty)
    (AtGrasp ?o ?g)
    (AtPose ?o ?p)
    (AtContact ?o1 ?o2 ?c)
    (InContact ?o1 ?o2)
    (FreeObj ?o)
    (Held ?o)
    (NotHeavy ?o)
  )

  ; Move in free space while not holing anything
  (:action move_free
    :parameters (?q1 ?q2 ?t)
    :precondition (and (FreeMotion ?q1 ?t ?q2)
                       (AtConf ?q1)
                       (HandEmpty))
    :effect (and (AtConf ?q2)
                 (not (AtConf ?q1)))
  )

  ; Move while holding Object ?o in Grasp ?g
  (:action move_holding
    :parameters (?q1 ?q2 ?o ?g ?t)
    :precondition (and (HoldingMotion ?q1 ?t ?q2 ?o ?g)
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
                       (Clear ?ot)
                       (On ?ot ?ob)
                       ;(NotHeavy ?ot)
                       (FreeObj ?ot))
    :effect (and (AtGrasp ?ot ?g)
                 (AtConf ?q2)
                 (Held ?ot)
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
                       (AtPose ?ob ?pb)
                       (Clear ?ob))
    :effect (and (AtPose ?ot ?pt)
                 (HandEmpty)
                 (AtConf ?q2)
                 (not (AtConf ?q1))
                 (not (AtGrasp ?ot ?g))
                 (On ?ot ?ob)
                 (not (Clear ?ob)))
  )

  ; Make contact ?c between ?o1 being held by robot and ?o2 (at ?p2)
  (:action move_contact
    :parameters (?o1 ?g ?o2 ?p1 ?p2 ?c ?q1 ?q2 ?t)
    :precondition (and (MakeContactMotion ?o1 ?g ?o2 ?p1 ?c ?q1 ?q2 ?t)
                       (AtConf ?q1)
                       (AtPose ?o2 ?p1)
                       (AtGrasp ?o1 ?g)
                       (Held ?o1)
                       (FreeObj ?o2))
    :effect (and (AtPose ?o2 ?p2)
                 (AtConf ?q2)
                 (not (AtPose ?o2 ?p1))
                 (not (AtConf ?q1)))
  )
)
