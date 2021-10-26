; Domain Description
;
; This domain is for the ordered blocks domain. The agent thinks it can place
; any numbered block on any other numbered block, but the top block must equal
; the bottom block + 1.

(define (domain ordered-blocks-optimistic)
  (:requirements :strips :equality)
  (:predicates
    (Block ?b)
    (Table ?t)
    (On ?bt ?bb)
    (Clear ?b)
    (HeightTwo ?b)
    (HeightThree ?b)
    (HeightFour ?b)
    (HeightFive ?b)
    (HeightSix ?b)
    (HeightSeven ?b)
    (HeightEight ?b)
  )

  ; Pick block ?bt which is on top of ?ob and place on object ?bb
  (:action pickplace
    :parameters (?bt ?ob ?bb)
    :precondition (and (Clear ?bt)
                       (On ?bt ?ob)
                       (Block ?bt)
                       (Clear ?bb)
                       (Block ?bb))

    :effect (and (On ?bt ?bb)
                 (not (Clear ?bb))
                 (not (On ?bt ?ob)))
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
