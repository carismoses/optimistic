; Domain Description
;
; This domain is for the ordered blocks domain. The agent thinks it can place
; any numbered block on any other numbered block, but the top block must equal
; the bottom block + 1.
;
; There are two types of objects in this world: Blocks and the Table.

(define (domain ordered-blocks-optimistic)
  (:requirements :strips :equality)
  (:predicates
    (On ?bt ?bb)
    (Clear ?b)
    (OnTable ?b)
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
)
