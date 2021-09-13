; Domain Description
;
; This domain is for the ordered blocks domain. The agent thinks it can place
; any numbered block on any other numbered block, but the top block must equal
; the bottom block + 1.

(define (domain ordered-blocks-learned)
  (:requirements :strips :equality)
  (:predicates
    (Block ?b)
    (On ?bt ?bb)
    (Clear ?b)
    (OnTable ?b)
    (TrustModel ?bt ?bb)
  )

  (:action stack
    :parameters (?bt ?bb)
    :precondition (and (Clear ?bb)
                       (Clear ?bt)
                       (OnTable ?bt)
                       (TrustModel ?bt ?bb))

    :effect (and (On ?bt ?bb)
                 (not (Clear ?bb))
                 (not (OnTable ?bt)))
  )
)
