package scienceworld.aer

import scienceworld.properties.MoveableProperties
import scienceworld.struct.EnvObject


/**
  * Retired seven-world prototype object.
  *
  * The active AER pea task never instantiates this class. It remains in source only so the
  * superseded counter-artifact implementation is auditable alongside its preserved trajectories.
  */
class AERPollinatorVisitCounter extends EnvObject {
  this.name = "pollinator visit counter"
  this.objType = "pollinator visit counter"
  this.propMoveable = Some(new MoveableProperties(isMovable = false))

  override def getReferents(): Set[String] = {
    Set(
      this.name,
      "visit counter",
      "pollinator counter",
      "optical counter",
      this.getDescriptName()
    )
  }

  override def getDescription(mode: Int): String = {
    "a retired pollinator visit counter with no active display"
  }
}
