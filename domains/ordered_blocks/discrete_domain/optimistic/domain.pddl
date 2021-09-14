; Domain Description
;
; This domain is for the ordered blocks domain. The agent thinks it can place
; any numbered block on any other numbered block, but the top block must equal
; the bottom block + 1.

(define (domain ordered-blocks-optimistic)
  (:requirements :strips :equality)
  (:predicates
    (On ?bt ?bb)
    (Clear ?b)
    (OnTable ?b)
    (HeightTwo ?b)
    (HeightThree ?b)
    (HeightFour ?b)
    (HeightFive ?b)
    (HeightSix ?b)
    (HeightSeven ?b)
    (HeightEight ?b)
  )

  (:action stack
    :parameters (?bt ?bb)
    :precondition (and (Clear ?bb)
                       (Clear ?bt)
                       (OnTable ?bt))

    :effect (and (On ?bt ?bb)
                 (not (Clear ?bb))
                 (not (OnTable ?bt)))
  )

  (:derived (HeightTwo ?b2)
    (exists (?b1) (and (On ?b2 ?b1)
                        (OnTable ?b1)))
  )

  (:derived (HeightThree ?b3)
    (exists (?b1 ?b2) (and (On ?b3 ?b2)
                            (On ?b2 ?b1)
                            (OnTable ?b1)))
  )

  (:derived (HeightFour ?b4)
    (exists (?b1 ?b2 ?b3) (and (On ?b4 ?b3)
                                (On ?b3 ?b2)
                                (On ?b2 ?b1)
                                (OnTable ?b1)))
  )

  (:derived (HeightFive ?b5)
    (exists (?b1 ?b2 ?b3 ?b4) (and (On ?b5 ?b4)
                                    (On ?b4 ?b3)
                                    (On ?b3 ?b2)
                                    (On ?b2 ?b1)
                                    (OnTable ?b1)))
  )

  (:derived (HeightSix ?b6)
    (exists (?b1 ?b2 ?b3 ?b4 ?b5) (and (On ?b6 ?b5)
                                        (On ?b5 ?b4)
                                        (On ?b4 ?b3)
                                        (On ?b3 ?b2)
                                        (On ?b2 ?b1)
                                        (OnTable ?b1)))
  )

  (:derived (HeightSeven ?b7)
    (exists (?b1 ?b2 ?b3 ?b4 ?b5 ?b6) (and (On ?b7 ?b6)
                                            (On ?b6 ?b5)
                                            (On ?b5 ?b4)
                                            (On ?b4 ?b3)
                                            (On ?b3 ?b2)
                                            (On ?b2 ?b1)
                                            (OnTable ?b1)))
  )

  (:derived (HeightEight ?b8)
    (exists (?b1 ?b2 ?b3 ?b4 ?b5 ?b6 ?b7) (and (On ?b8 ?b7)
                                                (On ?b7 ?b6)
                                                (On ?b6 ?b5)
                                                (On ?b5 ?b4)
                                                (On ?b4 ?b3)
                                                (On ?b3 ?b2)
                                                (On ?b2 ?b1)
                                                (OnTable ?b1)))
  )

)
