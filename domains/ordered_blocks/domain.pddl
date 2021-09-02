; Domain Description
;
; This domain is for the ordered blocks domain. The agent thinks it can place
; any numbered block on any other numbered block, but the top block must equal
; the bottom block + 1.
;
; There are two types of objects in this world: Blocks and the Table.
; Blocks can be moved but the Table is fixed.

(define (domain ordered-blocks)
  (:requirements :strips :equality)
  (:predicates
    (OnTable ?b)
    (OnTop ?b)
    (NoStack)
  )

  (:action stack
    :parameters (?bt ?bb)
    :precondition (and (OnTable ?bt)
                       (or (NoStack) (OnTop ?bb)))
    :effect (and (OnTop ?bt)
                 (not (OnTable ?bt))
                 (not (OnTop ?bb)))
  )

  (:derived (NoStack)
    (forall (?b) (OnTable ?b))
  )

)
