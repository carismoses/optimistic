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
    (Grasp ?g)
    (Conf ?q)
    (Pose ?b ?p)

    (FreeMotion ?q1 ?t ?q1)
    (HoldingMotion ?q1 ?t ?q2 ?b ?g)
    (PickKin ?b ?p ?g ?q1 ?q2 ?t)
    (PlaceKin ?bt ?pt ?g ?q1 ?q2 ?t)

    ; Fluents
    (On ?bt ?bb)
    (Clear ?b)
    (OnTable ?b)
    (AtConf ?q1)
    (HandEmpty)
    (AtGrasp ?b ?g)
    (AtPose ?b ?p)
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

  ; Pick up Block ?b at Pose ?p using Grasp ?g from the table
  (:action pick
    :parameters (?b ?p ?g ?q1 ?q2 ?t)
    :precondition (and (PickKin ?b ?p ?g ?q1 ?q2 ?t)
                       (AtPose ?b ?p)
                       (Clear ?b)
                       (OnTable ?b)
                       (HandEmpty)
                       (AtConf ?q1))
    :effect (and (AtGrasp ?b ?g)
                 (AtConf ?q2)
                 (not (AtConf ?q1))
                 (not (AtPose ?b ?p))
                 (not (HandEmpty))
                 (not (OnTable ?b)))
  )

  ; Place Block ?bt Block ?bb
  (:action place
    :parameters (?bt ?pt ?bb ?pb ?g ?q1 ?q2 ?t)
    :precondition (and (PlaceKin ?bt ?pt ?g ?q1 ?q2 ?t)
                       (Clear ?bb)
                       (AtGrasp ?bt ?g)
                       (AtConf ?q1)
                       (AtPose ?bb ?pb))
    :effect (and (AtPose ?bt ?pt)
                 (HandEmpty)
                 (AtConf ?q2)
                 (On ?bt ?bb)
                 (not (AtConf ?q1))
                 (not (AtGrasp ?bt ?g))
                 (not (Clear ?bb)))
  )
  )
