; Domain Description
;
; This domain is for the ordered blocks domain. The agent thinks it can place
; any numbered block on any other numbered block, but the top block must equal
; the bottom block + 1. This domain includes actions for the Panda robot.

(define (domain ordered-blocks-optimistic-panda)
  (:requirements :strips :equality)
  (:predicates
    ; Types
    (Block ?b)
    (Grasp ?b ?g)
    (Conf ?q)
    (Pose ?b ?p)
    (Table ?t)

    (FreeMotion ?q1 ?t ?q2)
    (HoldingMotion ?q1 ?t ?q2 ?b ?g)
    (PickKin ?b ?p ?g ?q1 ?q2 ?t)
    (PlaceKin ?b ?p ?g ?q1 ?q2 ?t)
    (Supported ?bt ?pt ?bb ?pb)

    (PickGraspKin ?b ?g ?q2)
    (PlaceGraspKin ?b ?g ?q2)

    ; Fluents
    (On ?bt ?bb)
    (Clear ?b)
    (AtConf ?q)
    (HandEmpty)
    (AtGrasp ?b ?g)
    (AtPose ?b ?p)

    (HeightTwo ?b)
    (HeightThree ?b)
    (HeightFour ?b)
    (HeightFive ?b)
    (HeightSix ?b)
    (HeightSeven ?b)
    (HeightEight ?b)
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

  ; Move while holding Block ?b in Grasp ?g
  (:action move_holding
    :parameters (?q1 ?q2 ?b ?g ?t)
    :precondition (and (HoldingMotion ?q1 ?t ?q2 ?b ?g)
                       (AtConf ?q1)
                       (AtGrasp ?b ?g))
    :effect (and (AtConf ?q2)
                 (not (AtConf ?q1)))
  )

  ; Pick up Block ?bt at Pose ?pt from Object (Block or Table) ?ob
  (:action pick
    :parameters (?bt ?pt ?ob ?g ?q1 ?q2 ?t)
    :precondition (and (PickKin ?bt ?pt ?g ?q1 ?q2 ?t)
                       (AtPose ?bt ?pt)
                       (HandEmpty)
                       (AtConf ?q1)
                       (Clear ?bt)
                       (On ?bt ?ob)
                       (Block ?bt))
    :effect (and (AtGrasp ?bt ?g)
                 (AtConf ?q2)
                 (not (AtConf ?q1))
                 (not (AtPose ?bt ?pt))
                 (not (HandEmpty))
                 (not (On ?bt ?ob)))
  )

  ; Place Block ?bt at Pose ?pt on Block ?bb which is at Pose ?pb (?bb is never a Table)
  (:action place
    :parameters (?bt ?pt ?bb ?pb ?g ?q1 ?q2 ?t)
    :precondition (and (PlaceKin ?bt ?pt ?g ?q1 ?q2 ?t)
                       (AtGrasp ?bt ?g)
                       (AtConf ?q1)
                       (Supported ?bt ?pt ?bb ?pb)
                       (AtPose ?bb ?pb)
                       (Clear ?bb)
                       (Block ?bt)
                       (Block ?bb))
    :effect (and (AtPose ?bt ?pt)
                 (HandEmpty)
                 (AtConf ?q2)
                 (not (AtConf ?q1))
                 (not (AtGrasp ?bt ?g))
                 (On ?bt ?bb)
                 (not (Clear ?bb)))
  )

  (:derived (HeightTwo ?b2)
    (exists (?t ?b1) (and (On ?b2 ?b1)
                       (On ?b1 ?t)
                       (Table ?t)
                       (Block ?b1)
                       (Block ?b2)))
  )

  (:derived (HeightThree ?b3)
    (exists (?t ?b1 ?b2) (and (On ?b3 ?b2)
                            (On ?b2 ?b1)
                            (On ?b1 ?t)
                            (Table ?t)
                            (Block ?b1)
                            (Block ?b2)
                            (Block ?b3)))
  )

  (:derived (HeightFour ?b4)
    (exists (?t ?b1 ?b2 ?b3) (and (On ?b4 ?b3)
                                (On ?b3 ?b2)
                                (On ?b2 ?b1)
                                (On ?b1 ?t)
                                (Table ?t)
                                (Block ?b1)
                                (Block ?b2)
                                (Block ?b3)
                                (Block ?b4)))
  )

  (:derived (HeightFive ?b5)
    (exists (?t ?b1 ?b2 ?b3 ?b4) (and (On ?b5 ?b4)
                                    (On ?b4 ?b3)
                                    (On ?b3 ?b2)
                                    (On ?b2 ?b1)
                                    (On ?b1 ?t)
                                    (Table ?t)
                                    (Block ?b1)
                                    (Block ?b2)
                                    (Block ?b3)
                                    (Block ?b4)
                                    (Block ?b5)))
  )

  (:derived (HeightSix ?b6)
    (exists (?t ?b1 ?b2 ?b3 ?b4 ?b5) (and (On ?b6 ?b5)
                                        (On ?b5 ?b4)
                                        (On ?b4 ?b3)
                                        (On ?b3 ?b2)
                                        (On ?b2 ?b1)
                                        (On ?b1 ?t)
                                        (Table ?t)
                                        (Block ?b1)
                                        (Block ?b2)
                                        (Block ?b3)
                                        (Block ?b4)
                                        (Block ?b5)
                                        (Block ?b6)))
  )

  (:derived (HeightSeven ?b7)
    (exists (?t ?b1 ?b2 ?b3 ?b4 ?b5 ?b6) (and (On ?b7 ?b6)
                                            (On ?b6 ?b5)
                                            (On ?b5 ?b4)
                                            (On ?b4 ?b3)
                                            (On ?b3 ?b2)
                                            (On ?b2 ?b1)
                                            (On ?b1 ?t)
                                            (Table ?t)
                                            (Block ?b1)
                                            (Block ?b2)
                                            (Block ?b3)
                                            (Block ?b4)
                                            (Block ?b5)
                                            (Block ?b6)
                                            (Block ?b7)))
  )

  (:derived (HeightEight ?b8)
    (exists (?t ?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7) (and (On ?b8 ?b7)
                                                (On ?b7 ?b6)
                                                (On ?b6 ?b5)
                                                (On ?b5 ?b4)
                                                (On ?b4 ?b3)
                                                (On ?b3 ?b2)
                                                (On ?b2 ?b1)
                                                (On ?b1 ?t)
                                                (Table ?t)
                                                (Block ?b1)
                                                (Block ?b2)
                                                (Block ?b3)
                                                (Block ?b4)
                                                (Block ?b5)
                                                (Block ?b6)
                                                (Block ?b7)
                                                (Block ?b8)))
  )
)
